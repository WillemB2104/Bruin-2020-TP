import os
from time import time

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import distributions
from shogun import RealFeatures, BinaryLabels, LinearKernel, SqrtDiagKernelNormalizer, ConstMean, \
    LogitLikelihood, SingleLaplaceInferenceMethod, GaussianProcessClassification
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_handling import create_data_set

CAT_COV = ['age_group', 'tesla', 'Agr_Check', 'Anx', 'Clean', 'CurAnx', 'CurDep', 'Dep', 'Hoard', 'Med', 'Ord', 'Sex',
           'Sex_Rel', 'age_group_tesla_site']


def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Ref: http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.lstsq(A[M], B[M])[0]

    # Else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:, :, None]  # n x r x 1 tensor
    T = np.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # n x r x r tensor

    return np.squeeze(np.linalg.solve(T, rhs)).T  # transpose to get r x n


def auc2p(auc_score, n1, n2):
    # Convert AUC score to P value using Mann–Whitney U test (Wilcoxon rank-sum) statistic and normal distribution

    # Transform AUC into U stat|
    u1 = n1 * n2 * (1 - auc_score)
    u2 = n1 * n2 * auc_score

    bigu = u2  # Alternative == greater; AUC was calculated with 1 as positive class

    # Standardize U
    meanrank = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (bigu - meanrank) / sd

    # Convert Z to P
    return u1, u2, distributions.norm.sf(z)


def roc_auc_ci(auc_score, n1, n2):
    # Hanley-McNeil’s Method.

    # AUC = roc_auc_score(y_true, y_score)
    # n1 = sum(y_true == positive)
    # n2 = sum(y_true != positive)

    q1 = auc_score / (2 - auc_score)
    q2 = 2 * auc_score ** 2 / (1 + auc_score)

    # Mann–Whitney U test / Wilcoxon rank-sum test approximation to estimate standard error of ROC-AUC
    se_auc = np.sqrt((auc_score * (1 - auc_score) + (n1 - 1) * (q1 - auc_score ** 2) + (n2 - 1) * (q2 - auc_score ** 2))
                     / (n1 * n2))

    # Calculate 95% confidence intervals
    lower = auc_score - 1.96 * se_auc
    upper = auc_score + 1.96 * se_auc
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return lower, upper


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def create_error_str(site_wise_norm_mask, string_p1, column_type, empty_C, empty_R, empty_columns, column_labels):
    error_str = "\n".join([column_type + ": " + str(column_labels[site_wise_norm_mask][fs_C]) + string_p1 + str(
        empty_R[empty_C == fs_C]) for fs_C in np.unique(empty_C)]) if np.any(empty_columns) else ''
    return error_str


def nan_matrix_site_columns(X, mask, g_counts, g_vals, groups):
    nan_matrix = np.array(
        [np.sum(np.isnan(X[:, mask][groups == val]), axis=0) == count for val, count in
         zip(g_vals, g_counts)]) if np.any(mask) else np.array([]).reshape((len(g_vals), 0))
    return nan_matrix


def assert_mask(X, mask, mask_name):
    if not isinstance(mask, list) and not isinstance(mask, np.ndarray):
        raise RuntimeError(
            "Error: mask " + mask_name + " has to be either a list or np.ndarray, type {} not accepted".format(
                type(mask)))
    elif X.shape[-1] != len(mask) or np.array(mask).dtype != bool:
        raise RuntimeError(
            "Error: mask " + mask_name + " has to match length {} and dtype bool,\n"
                                         "Given length {} and dtype {} not accepted".format(X.shape[-1], len(mask),
                                                                                            np.array(mask).dtype))


def extract_cv_label(cv):
    tmp = str(cv)
    return tmp.split('(')[0]


def get_random_state(obj):
    try:
        return obj.random_state
    except AttributeError:
        return None


def create_filename(C_size, _cov_sw_norm_mask, _cov_to_regress_mask, _fs_sw_norm_mask, covariates, inner_cv_label,
                    outer_cv_label, custom_suffix, kwargs):
    if len(_fs_sw_norm_mask) == sum(_fs_sw_norm_mask):
        fs_suffix = "FS_SWN_1*"
    elif sum(_fs_sw_norm_mask) == 0:
        fs_suffix = "FS_SWN_0*"
    else:
        fs_suffix = "FS_SWN_" + ''.join([str(int(x)) for x in _fs_sw_norm_mask])
    if C_size > 0:
        covariate_str_1 = "COV_" + '-'.join(covariates)
        covariate_str_2 = "_SWN_" + ''.join([str(int(x)) for x in _cov_sw_norm_mask])
        covariate_str_3 = "_REG_" + ''.join([str(int(x)) for x in _cov_to_regress_mask])
        cov_suffix = covariate_str_1 + covariate_str_2 + covariate_str_3
    else:
        cov_suffix = "COV_None"
    age_group = kwargs.get("age_group") if "age_group" in kwargs.keys() else None
    age_suffix = age_group + "_" if age_group else "3_combined_"
    cov_only = kwargs.get("use_covariates_only")
    if cov_only:
        custom_suffix = custom_suffix + "_cov_only" if custom_suffix else "cov_only"
    filename_prefix = outer_cv_label + "_" + inner_cv_label + "_" + fs_suffix + "_" + cov_suffix + "_" + age_suffix
    if custom_suffix:
        filename_prefix = filename_prefix + custom_suffix + "_"
    disable_imputation = kwargs.get("complete")
    if disable_imputation:
        filename_prefix = filename_prefix + "non_imputed_"
    return filename_prefix


def create_normalization_masks(X, _fs_sw_norm, _fs_sw_norm_mask,
                               C, _cov_sw_norm, _cov_sw_norm_mask, _cov_to_regress_mask, kwargs):
    if _fs_sw_norm_mask:
        assert_mask(X, _fs_sw_norm_mask, "_fs_sw_norm_mask")
    else:
        _fs_sw_norm_mask = np.ones(X.shape[-1], dtype=bool) if _fs_sw_norm else np.zeros(X.shape[-1], dtype=bool)

    if 'covariates' in kwargs and C.size > 0:
        covariates = np.array(kwargs.get("covariates"))
        _cat_cov_mask = np.array([c in CAT_COV for c in covariates], dtype=bool)
        if _cov_sw_norm_mask:
            assert_mask(C, _cov_sw_norm_mask, "_cov_sw_norm_mask")
        else:
            _cov_sw_norm_mask = np.ones(C.shape[-1], dtype=bool) if _cov_sw_norm else np.zeros(C.shape[-1], dtype=bool)
        if _cov_to_regress_mask:
            assert_mask(C, _cov_to_regress_mask, "_cov_to_regress_mask")
        else:
            _cov_to_regress_mask = np.zeros(C.shape[-1], dtype=bool)

        # Create new masks to index categorical (cat) and non-cat covariates that have to be normalized (and imputed)
        # either per-site or on total sample
        _cat_cov_sw_norm = _cov_sw_norm_mask * _cat_cov_mask
        _non_cat_cov_sw_norm = _cov_sw_norm_mask * np.invert(_cat_cov_mask)
        _cat_cov_tot_norm = np.invert(_cov_sw_norm_mask) * _cat_cov_mask
        _non_cat_cov_tot_norm = np.invert(_cov_sw_norm_mask) * np.invert(_cat_cov_mask)
    else:
        covariates = None
        _cat_cov_mask = None
        _cov_sw_norm_mask = np.zeros(C.shape[-1], dtype=bool)
        _cat_cov_sw_norm = _non_cat_cov_sw_norm = _cat_cov_tot_norm = _non_cat_cov_tot_norm = np.array([], dtype=bool)

    # Create mask of FS features + non-cat covariates to normalize and impute per site
    cols_to_norm_and_imp_per_site = np.append(_fs_sw_norm_mask, _non_cat_cov_sw_norm)

    # Create mask of categorical covariates as these have to be only imputed per site (no centering and scaling)
    cols_to_imp_per_site = np.append(np.zeros(X.shape[-1], dtype=bool), _cat_cov_sw_norm)

    # Do the same for columns that have to be imputed (and normalized) over total sample
    cols_to_norm_and_imp_on_total = np.append(np.invert(_fs_sw_norm_mask), _non_cat_cov_tot_norm)
    cols_to_imp_on_total = np.append(np.zeros(X.shape[-1], dtype=bool), _cat_cov_tot_norm)

    return _cat_cov_mask, _cov_sw_norm_mask, _cov_to_regress_mask, _fs_sw_norm_mask, cols_to_imp_on_total, \
           cols_to_imp_per_site, cols_to_norm_and_imp_on_total, cols_to_norm_and_imp_per_site, covariates


def site_wise_normalization(X, C, _cov_site_wise_norm_mask, _fs_site_wise_norm_mask, cols_to_imp_per_site,
                            cols_to_norm_and_imp_per_site, covariates, fs_labels, groups, train_id, test_id,
                            logo_cv=False):
    X_C = np.c_[X, C]
    X_C_train, X_C_test = X_C[train_id], X_C[test_id]

    groups_train, groups_test = groups[train_id], groups[test_id]
    g_train_val, g_train_counts = np.unique(groups_train, return_counts=True)
    group_vals = np.unique(groups_test)

    # Calculate the number of NaNs per column in training data
    fs_nans = nan_matrix_site_columns(X[train_id], _fs_site_wise_norm_mask, g_train_counts, g_train_val,
                                      groups_train)
    cov_nans = nan_matrix_site_columns(C[train_id], _cov_site_wise_norm_mask, g_train_counts, g_train_val,
                                       groups_train)
    empty_fs_columns, empty_cov_columns = np.any(fs_nans, axis=0), np.any(cov_nans, axis=0)

    # Verify whether site-wise scaling is possible (there can't be any empty columns)
    if np.any(empty_fs_columns) or np.any(empty_cov_columns):
        empty_fs_R, empty_fs_C = np.where(fs_nans)
        empty_cov_R, empty_cov_C = np.where(cov_nans)
        string_p1 = " missing for all participants in site(s): "
        err_str_1 = create_error_str(_fs_site_wise_norm_mask, string_p1, "FS feature(s)", empty_fs_C,
                                     empty_fs_R, empty_fs_columns, fs_labels)
        err_str_2 = create_error_str(_cov_site_wise_norm_mask, string_p1, "Covariate(s)", empty_cov_C,
                                     empty_cov_R, empty_cov_columns, covariates)
        raise RuntimeError(
            "Warning! Unable to perform site-wise scaling: \n{} \n"
            "Consider removing these site(s), or to normalise columns on total sample instead".format(
                err_str_1 + err_str_2))

    # Empty matrices to store site-normalized data in
    X_C_train_to_norm_and_imp_per_site = np.zeros_like(X_C_train[:, cols_to_norm_and_imp_per_site])
    cat_cov_train_to_imp_per_site = np.zeros_like(X_C_train[:, cols_to_imp_per_site]) if np.any(
        cols_to_imp_per_site) else None

    if logo_cv:
        return site_wise_logo_normalization(X_C_test, X_C_train, X_C_train_to_norm_and_imp_per_site,
                                            cat_cov_train_to_imp_per_site, cols_to_imp_per_site,
                                            cols_to_norm_and_imp_per_site, groups_train)

    X_C_test_to_norm_and_imp_per_site = np.zeros_like(X_C_test[:, cols_to_norm_and_imp_per_site])
    cat_cov_test_to_imp_per_site = np.zeros_like(X_C_test[:, cols_to_imp_per_site]) if np.any(
        cols_to_imp_per_site) else None

    # Empty matrices to store center's and scales per site
    center_per_site = np.zeros((len(group_vals), sum(cols_to_norm_and_imp_per_site)))
    scale_per_site = np.zeros((len(group_vals), sum(cols_to_norm_and_imp_per_site)))
    center_cat_cov_per_site = np.zeros((len(group_vals), sum(cols_to_imp_per_site))) if np.any(
        cols_to_imp_per_site) else None

    # Iterate over sites and perform normalization
    for i_g, group in enumerate(group_vals):
        tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

        if np.any(cols_to_norm_and_imp_per_site):
            # Split up in X_train and X_test here
            X_C_train_group = X_C_train[:, cols_to_norm_and_imp_per_site][groups_train == group]
            X_C_test_group = X_C_test[:, cols_to_norm_and_imp_per_site][groups_test == group]

            # 1. Estimate median (using training data only!)
            tmp_imputer.fit(X_C_train_group)
            center_ = tmp_imputer.statistics_

            # 2. Estimate interquartile range (using training data only!)
            q_min, q_max = 25.0, 75.0
            q = np.nanpercentile(X_C_train_group, [q_min, q_max], axis=0)
            scale_ = (q[1] - q[0])
            scale_ = _handle_zeros_in_scale(scale_, copy=False)

            # 3. Impute median
            X_C_train_group = tmp_imputer.transform(X_C_train_group)
            X_C_test_group = tmp_imputer.transform(X_C_test_group)

            # 4. Robust scale
            X_C_train_group -= center_
            X_C_train_group /= scale_
            X_C_test_group -= center_
            X_C_test_group /= scale_

            # 5. Store normalised matrices along with each site's center and scale for later usage
            center_per_site[i_g, :] = center_
            scale_per_site[i_g, :] = scale_

            X_C_train_to_norm_and_imp_per_site[groups_train == group] = X_C_train_group
            X_C_test_to_norm_and_imp_per_site[groups_test == group] = X_C_test_group

        # Check if there are any cat covariates to center per site
        if np.any(cols_to_imp_per_site):
            cat_cov_train_group = X_C_train[:, cols_to_imp_per_site][groups_train == group]
            cat_cov_test_group = X_C_test[:, cols_to_imp_per_site][groups_test == group]
            cat_cov_train_group = tmp_imputer.fit_transform(cat_cov_train_group)
            cat_cov_test_group = tmp_imputer.transform(cat_cov_test_group)

            cat_cov_train_to_imp_per_site[groups_train == group] = cat_cov_train_group
            cat_cov_test_to_imp_per_site[groups_test == group] = cat_cov_test_group
            center_cat_cov_per_site[i_g, :] = tmp_imputer.statistics_

    X_C_train[:, cols_to_norm_and_imp_per_site] = X_C_train_to_norm_and_imp_per_site
    X_C_test[:, cols_to_norm_and_imp_per_site] = X_C_test_to_norm_and_imp_per_site
    X_C_train[:, cols_to_imp_per_site] = cat_cov_train_to_imp_per_site
    X_C_test[:, cols_to_imp_per_site] = cat_cov_test_to_imp_per_site

    return X_C_train, X_C_test


def site_wise_logo_normalization(X_C_test, X_C_train, X_C_train_to_norm_and_imp_per_site, cat_cov_train_to_imp_per_site,
                                 cols_to_imp_per_site, cols_to_norm_and_imp_per_site, groups_train):
    group_vals = np.unique(groups_train)

    for i_g, group in enumerate(group_vals):

        # Split up in X_train and X_test here
        X_C_train_group = X_C_train[:, cols_to_norm_and_imp_per_site][groups_train == group]

        # 1. Estimate median (using training data only!)
        tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        tmp_imputer.fit(X_C_train_group)
        center_ = tmp_imputer.statistics_

        # 2. Estimate interquartile range (using training data only!)
        q_min, q_max = 25.0, 75.0
        q = np.nanpercentile(X_C_train_group, [q_min, q_max], axis=0)
        scale_ = (q[1] - q[0])
        scale_ = _handle_zeros_in_scale(scale_, copy=False)

        # 3. Impute median
        X_C_train_group = tmp_imputer.transform(X_C_train_group)

        # Check if there are any cat covariates to center per site
        if np.any(cols_to_imp_per_site):
            cat_cov_train_group = X_C_train[:, cols_to_imp_per_site][groups_train == group]
            cat_cov_train_group = tmp_imputer.fit_transform(cat_cov_train_group)

            cat_cov_train_to_imp_per_site[groups_train == group] = cat_cov_train_group

        # 4. Robust scale
        X_C_train_group -= center_
        X_C_train_group /= scale_

        X_C_train_to_norm_and_imp_per_site[groups_train == group] = X_C_train_group

    X_C_test_group = X_C_test[:, cols_to_norm_and_imp_per_site]
    tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    tmp_imputer.fit(X_C_test_group)
    center_ = tmp_imputer.statistics_
    q_min, q_max = 25.0, 75.0
    q = np.nanpercentile(X_C_test_group, [q_min, q_max], axis=0)
    scale_ = (q[1] - q[0])
    scale_ = _handle_zeros_in_scale(scale_, copy=False)
    X_C_test_group = tmp_imputer.transform(X_C_test_group)

    X_C_test_group -= center_
    X_C_test_group /= scale_

    # Write normalized columns out to train and test data
    X_C_train[:, cols_to_norm_and_imp_per_site] = X_C_train_to_norm_and_imp_per_site
    X_C_test[:, cols_to_norm_and_imp_per_site] = X_C_test_group

    # Check if there are any cat covariates to center per site
    if np.any(cols_to_imp_per_site):
        X_C_train[:, cols_to_imp_per_site] = cat_cov_train_to_imp_per_site

        cat_cov_test_group = X_C_test[:, cols_to_imp_per_site]
        cat_cov_test_group = tmp_imputer.fit_transform(cat_cov_test_group)
        X_C_test[:, cols_to_imp_per_site] = cat_cov_test_group

    return X_C_train, X_C_test


def total_normalization(X_C_train, X_C_test, cols_to_imp_on_total, cols_to_norm_and_imp_on_total):
    if np.any(cols_to_norm_and_imp_on_total):
        X_C_train_to_norm_on_tot = X_C_train[:, cols_to_norm_and_imp_on_total]
        X_C_test_to_norm_on_tot = X_C_test[:, cols_to_norm_and_imp_on_total]

        tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        tmp_imputer.fit(X_C_train_to_norm_on_tot)
        center_ = tmp_imputer.statistics_

        q_min, q_max = 25.0, 75.0
        q = np.nanpercentile(X_C_train_to_norm_on_tot, [q_min, q_max], axis=0)
        scale_ = (q[1] - q[0])
        scale_ = _handle_zeros_in_scale(scale_, copy=False)

        X_C_train_to_norm_on_tot = tmp_imputer.transform(X_C_train_to_norm_on_tot)
        X_C_test_to_norm_on_tot = tmp_imputer.transform(X_C_test_to_norm_on_tot)

        X_C_train_to_norm_on_tot -= center_
        X_C_train_to_norm_on_tot /= scale_
        X_C_test_to_norm_on_tot -= center_
        X_C_test_to_norm_on_tot /= scale_

        X_C_train[:, cols_to_norm_and_imp_on_total] = X_C_train_to_norm_on_tot
        X_C_test[:, cols_to_norm_and_imp_on_total] = X_C_test_to_norm_on_tot

    if np.any(cols_to_imp_on_total):
        cat_cov_train_to_norm_on_tot = X_C_train[:, cols_to_imp_on_total]
        cat_cov_test_to_norm_on_tot = X_C_test[:, cols_to_imp_on_total]

        tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        cat_cov_train_to_norm_on_tot = tmp_imputer.fit_transform(cat_cov_train_to_norm_on_tot)
        cat_cov_test_to_norm_on_tot = tmp_imputer.transform(cat_cov_test_to_norm_on_tot)

        X_C_train[:, cols_to_imp_on_total] = cat_cov_train_to_norm_on_tot
        X_C_test[:, cols_to_imp_on_total] = cat_cov_test_to_norm_on_tot

    return X_C_train, X_C_test


def encode_cat_2_dummy(X_C_test, X_C_train, _cat_cov_mask, _cov_to_regress_mask, covariates, X_C_labels,
                       test_id, train_id):
    tmp_X_C = np.zeros((len(X_C_train) + len(X_C_test), len(X_C_labels)))
    tmp_X_C[train_id] = X_C_train
    tmp_X_C[test_id] = X_C_test

    cat_column_mask = np.zeros(len(X_C_labels), dtype=bool)
    cat_column_mask[-len(covariates):] = _cat_cov_mask

    hot_encoded_cat_cov = pd.get_dummies(
        pd.DataFrame(data=tmp_X_C[:, cat_column_mask], columns=X_C_labels[cat_column_mask]),
        columns=X_C_labels[cat_column_mask], drop_first=True)

    dummy_labels, dummy_data = hot_encoded_cat_cov.columns.values, hot_encoded_cat_cov.values

    tmp_X_C = np.c_[tmp_X_C[:, ~cat_column_mask], dummy_data]
    X_C_train = tmp_X_C[train_id]
    X_C_test = tmp_X_C[test_id]

    dum_X_C_labels = np.append(X_C_labels[~cat_column_mask], dummy_labels)
    dum_covariates = np.append(covariates[~_cat_cov_mask], dummy_labels)

    dum_cov_to_regress_mask = np.append(np.array(_cov_to_regress_mask)[~_cat_cov_mask],
                                        [d.rsplit('_', 1)[0] in covariates[_cov_to_regress_mask] for d in
                                         dummy_labels])

    dum_cat_cov_mask = np.append(np.zeros_like(covariates[~_cat_cov_mask], dtype=bool),
                                 [d.rsplit('_', 1)[0] in CAT_COV for d in dummy_labels])

    return X_C_test, X_C_train, dum_cat_cov_mask, dum_cov_to_regress_mask, dum_covariates, dum_X_C_labels


def regress_out_cov(X, X_C_labels, X_C_test, X_C_train, _cov_to_regress_mask, covariates, train_id):
    # Extract confound matrices
    c_ = np.array([c in covariates[_cov_to_regress_mask] for c in X_C_labels])
    C_train = X_C_train[:, c_]
    C_test = X_C_test[:, c_]

    # Add intercept terms
    icept_train, icept_test = np.ones(C_train.shape[0]), np.ones(C_test.shape[0])
    confound_train, confound_test = np.c_[icept_train, C_train], np.c_[icept_test, C_test]

    # Extract features which will be regressed out (set to FS features only for now)
    fs_nan_idx = np.isnan(X[train_id])
    freesurfer_data_train = X_C_train[:, :len(X_C_labels) - len(covariates)]
    freesurfer_data_train[fs_nan_idx] = 0

    # Create tensor mask for NaNs
    M = (fs_nan_idx.astype(int) * -1) + 1
    censored_lstsq_solution = censored_lstsq(confound_train, freesurfer_data_train, M)

    # Find least square solution and remove regressors
    freesurfer_data_test = X_C_test[:, :len(X_C_labels) - len(covariates)]
    freesurfer_data_train_residuals = freesurfer_data_train - confound_train.dot(censored_lstsq_solution)
    freesurfer_data_test_residuals = freesurfer_data_test - confound_test.dot(censored_lstsq_solution)
    X_C_train[:, :len(X_C_labels) - len(covariates)] = freesurfer_data_train_residuals
    X_C_test[:, :len(X_C_labels) - len(covariates)] = freesurfer_data_test_residuals

    return X_C_test, X_C_train


def get_models_to_check():
    svm = Pipeline([('svm', SVC(probability=True, class_weight='balanced', kernel='linear', random_state=0))])
    pca_svm = Pipeline([('pca', PCA(n_components=0.9)),
                        ('svm', SVC(probability=True, class_weight='balanced', kernel='linear', random_state=0))
                        ])
    rbf_svm = Pipeline([('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=0))])
    pca_rbf_svm = Pipeline([('pca', PCA(n_components=0.9)),
                            ('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=0))
                            ])
    log = Pipeline([('logistic', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0))])
    pca_log = Pipeline([('pca', PCA(n_components=0.9)),
                        ('logistic', LogisticRegression(class_weight='balanced', solver='liblinear', random_state=0))])

    xgboost = Pipeline([('xgboost', XGBClassifier(eval_metric='auc', nthread=1, random_state=0))])

    clfs = [svm, pca_svm, rbf_svm, pca_rbf_svm, log, pca_log, xgboost]
    clfs_labels = ['svm', 'pca_svm', 'rbf_svm', 'pca_rbf_svm', 'log', 'pca_log', 'xgboost']
    return clfs, clfs_labels


def brfc_classification(X_train, y_train, X_test, n_jobs):
    y_vals, y_counts = np.unique(y_train, return_counts=True)
    n_minority_class = y_counts[np.argsort(y_counts)][0]
    subsample_size = int(round((n_minority_class * 0.5)))
    brfc = BalancedRandomForestClassifier(n_estimators=1000, class_weight="balanced", oob_score=False,
                                          sampling_strategy={0: subsample_size, 1: subsample_size},
                                          n_jobs=n_jobs, random_state=0, bootstrap=False, replacement=False)
    t1 = time()
    brfc.fit(X_train, y_train)
    t2 = time()
    print('Fitted bRFC in {:.2f} minutes \n'.format((t2 - t1) / 60.))
    y_pred = brfc.predict(X_test)
    y_score = brfc.predict_proba(X_test)[:, 1]
    return brfc, y_pred, y_score


def gpc_classification(X_train, y_train, X_test, y_test):
    """Perform gaussian processes classification using shogun.

    Returns fitted classifier, class predictions and probabilities given features and labels of test and train data.
    """

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    X_train = X_train.T
    X_test = X_test.T

    features_train = RealFeatures(X_train)
    features_test = RealFeatures(X_test)
    labels_train = BinaryLabels(y_train)

    kernel = LinearKernel()
    kernel.set_normalizer(SqrtDiagKernelNormalizer())
    mean_function = ConstMean()

    gauss_likelihood = LogitLikelihood()
    inference_method = SingleLaplaceInferenceMethod(kernel, features_train, mean_function, labels_train,
                                                    gauss_likelihood)

    gp_classifier = GaussianProcessClassification(inference_method)

    t1 = time()
    gp_classifier.train()
    t2 = time()

    print('Finished training GP classifier in {:.2f} minutes \n'.format((t2 - t1) / 60.))

    labels_predict = gp_classifier.apply_binary(features_test)
    y_score = gp_classifier.get_probabilities(features_test)

    if np.any(np.isnan(y_score)):
        print(
            "Warning, nan detected in score")  # Only happens when all features are set to 0. Only the case in adult MED clf with covariates onl
        y_score[np.isnan(y_score)] = 0.5

    y_pred = labels_predict.get_labels()
    y_pred[y_pred == -1] = 0

    return gp_classifier, y_pred, y_score


def inner_loop_iteration(clf, clf_label, X, y, groups, cv, n_jobs):
    if clf_label == 'rfc':
        grid_params = {'rfc__n_estimators': np.linspace(10, 200, 5, dtype=int),  # Should be as HIGH as possible
                       'rfc__max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75]}
    elif 'svm' in clf_label:
        grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1]}
        if 'rbf' in clf_label:
            grid_params = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1],
                           'svm__gamma': [0.001, 0.01, 0.1, 1]}
    elif 'log' in clf_label:
        grid_params = {'logistic__C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                       'logistic__penalty': ['l1', 'l2']}
    elif 'xgboost' in clf_label:
        grid_params = {'xgboost__gamma': [0.5, 1, 1.5, 2, 5],
                       'xgboost__learning_rate': [0.01, 0.1, 0.3],
                       'xgboost__max_depth': [3, 4, 5]}
    else:
        return clf, None
    grid_search = GridSearchCV(estimator=clf, param_grid=grid_params, scoring='roc_auc', cv=cv, refit=True, verbose=1,
                               n_jobs=n_jobs, return_train_score=False, pre_dispatch='2*n_jobs')
    grid_search.fit(X, y, groups)
    best_clf = grid_search.best_estimator_
    results_df = pd.DataFrame(grid_search.cv_results_)
    return best_clf, results_df


def str_to_list(argument_string, dtype):
    if argument_string:
        tmp = argument_string.rsplit(',')
        if dtype == bool:
            tmp = [arg == 'True' for arg in tmp]
        return [*map(dtype, tmp)]


def create_data_from_filename(filename, results_path):
    if "_clf_models.npy" in filename:
        trimmed_filename = filename.split('_clf_models.npy')[0]
    elif "_roc_curves.npy" in filename:
        trimmed_filename = filename.split('_roc_curves.npy')[0]
    elif "_predictions.npy" in filename:
        trimmed_filename = filename.split('_predictions.npy')[0]
    elif "_performance_metrics.npz" in filename:
        trimmed_filename = filename.split('_performance_metrics.npz')[0]
    elif "_arguments.npy" in filename:
        trimmed_filename = filename.split('_arguments.npy')[0]
    elif "_perm_importances_brfc.csv" in filename:
        trimmed_filename = filename.split('_perm_importances_brfc.csv')[0]
    else:
        return

    arguments_file = trimmed_filename + "_arguments.npy"
    models_file = trimmed_filename + "_clf_models.npy"
    if os.path.exists(os.path.join(results_path, arguments_file)):
        kwargs = np.load(os.path.join(results_path, arguments_file), allow_pickle=True)
    else:
        return "Arguments not found!"
    if os.path.exists(os.path.join(results_path, arguments_file)):
        print("Loading classifier models")
        models = np.load(os.path.join(results_path, models_file), allow_pickle=True)
    else:
        return "Models not found!"

    kwargs = kwargs.tolist()
    allowed_args = create_data_set.__code__.co_varnames
    filtered_kwargs = {d: kwargs[d] for d in kwargs if d in allowed_args}
    covariates = np.array(filtered_kwargs['covariates'])
    y_label = filtered_kwargs['y_label']
    filtered_kwargs['verbose'] = False
    X, selected_fs_features, C, y, groups = create_data_set(**filtered_kwargs)
    return X, selected_fs_features, C, covariates, y, y_label, groups, models, trimmed_filename, kwargs
