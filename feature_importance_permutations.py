#!/usr/bin/env python
# coding: utf-8

import os
from os import path as osp
from time import time

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.impute import SimpleImputer

from data_handling import ensure_folder
from helper_functions import _handle_zeros_in_scale, CAT_COV, create_data_from_filename
from run_between_site_clf import CLF_LABELS


def permutation_test(neutral_permutation, permutation_values):
    return (np.sum(permutation_values >= neutral_permutation) + 1.) / (permutation_values.size + 1.)


def permute_feature(estimator, X, y, feature_id=0):
    X[:, feature_id] = np.random.permutation(X[:, feature_id])
    estimator.fit(X, y)
    feat_imp_perm = estimator.feature_importances_
    return feat_imp_perm[feature_id]


def dummy_encode_labels(X, C, covariates, fs_labels):
    if C.size > 0:
        X_C = np.c_[X, C]
        X_C_labels = np.append(fs_labels, covariates) if C.size > 0 else fs_labels  # prevent appending None's

        _cat_cov_mask = np.array([c in CAT_COV for c in covariates], dtype=bool)
        cat_column_mask = np.zeros(len(X_C_labels), dtype=bool)
        cat_column_mask[-len(covariates):] = _cat_cov_mask

        hot_encoded_cat_cov = pd.get_dummies(
            pd.DataFrame(data=X_C[:, cat_column_mask], columns=X_C_labels[cat_column_mask]),
            columns=X_C_labels[cat_column_mask], drop_first=True)

        dummy_labels, dummy_data = hot_encoded_cat_cov.columns.values, hot_encoded_cat_cov.values

        tmp_X_C = np.c_[X_C[:, ~cat_column_mask], dummy_data]
        dum_X_C_labels = np.append(X_C_labels[~cat_column_mask], dummy_labels)
        dum_covariates = np.append(covariates[~_cat_cov_mask], dummy_labels)
        return tmp_X_C, dum_X_C_labels, dum_covariates
    else:
        return X, fs_labels, covariates


def run(result):
    X, selected_fs_features, C, covariates, y, y_label, groups, models, trimmed_filename, kwargs = \
        create_data_from_filename(result, RESULTS_DIR)

    if 'custom_suffix' in kwargs:
        if 'all_cov_regressed' in kwargs['custom_suffix']:
            print("Dropping covariates {}".format(np.array(covariates)))
            C = np.zeros((X.shape[0], 0))

    # Prepare covariates (if any)
    if C.size > 0:
        _cat_cov_mask = np.array([c not in CAT_COV for c in covariates], dtype=bool)
        tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
        tmp_imputer.fit(C)
        center_ = tmp_imputer.statistics_
        q_min, q_max = 25.0, 75.0
        q = np.nanpercentile(C, [q_min, q_max], axis=0)
        scale_ = (q[1] - q[0])
        scale_ = _handle_zeros_in_scale(scale_, copy=False)
        C = tmp_imputer.transform(C)
        if np.any(_cat_cov_mask):
            print("Normalising {}".format(np.array(covariates)[_cat_cov_mask]))
            C[:, _cat_cov_mask] -= center_[_cat_cov_mask]
            C[:, _cat_cov_mask] /= scale_[_cat_cov_mask]

    # Prepare X
    tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    tmp_imputer.fit(X)
    center_ = tmp_imputer.statistics_
    q_min, q_max = 25.0, 75.0
    q = np.nanpercentile(X, [q_min, q_max], axis=0)
    scale_ = (q[1] - q[0])
    scale_ = _handle_zeros_in_scale(scale_, copy=False)
    X = tmp_imputer.transform(X)
    X -= center_
    X /= scale_

    cv_iters = models.shape[-1]

    X_C, feature_labels, dummy_cov_labels = dummy_encode_labels(X, C, covariates, selected_fs_features)
    idx = np.append(["cv_" + str(i) for i in range(cv_iters)], ["perm_" + str(i) for i in range(N_PERM)])
    df_feat_imp = pd.DataFrame(index=idx, columns=feature_labels, dtype=np.float)

    for i, m in enumerate(models[BRFC_IDX]):
        if len(m.feature_importances_) is not len(feature_labels):
            raise RuntimeError(
                "Error: feature importance's length mismatch: got {}, expected {}\n".format(len(m.feature_importances_),
                                                                                            len(feature_labels)))
        df_feat_imp.iloc[i, :] = m.feature_importances_

    y_vals, y_counts = np.unique(y, return_counts=True)
    n_minority_class = y_counts[np.argsort(y_counts)][0]
    subsample_size = int(round((n_minority_class * 0.5)))

    # Run permutations
    estimator = BalancedRandomForestClassifier(n_estimators=1000, class_weight="balanced", oob_score=False,
                                               sampling_strategy={0: subsample_size, 1: subsample_size},
                                               n_jobs=N_JOBS_RF, random_state=1, bootstrap=False, replacement=False)

    # If covariates are used, check if there are any (dummy encoded) features that have to be block permuted
    if C.size > 0:

        first_cov_idx = len(feature_labels) - len(dummy_cov_labels)
        dummy_mask = [f.rsplit('_', 1)[0] for f in feature_labels[first_cov_idx:]]
        dummies, n_dummies = np.unique(dummy_mask, return_counts=True)
        to_perm_as_block_labels = dummies[n_dummies > 1]
        to_perm_as_block_idx = []

        # Create lists of indices that have to be permuted column wise
        for p in to_perm_as_block_labels:
            bool_block_mask = [d in p for d in dummy_mask]
            block_perm_idx = np.arange(X_C.shape[1])[first_cov_idx:][bool_block_mask]
            to_perm_as_block_idx.append(block_perm_idx)

        to_perm_columnwise = set(np.arange(X_C.shape[1])) - set([j for i in to_perm_as_block_idx for j in i])

        for i, block in enumerate(to_perm_as_block_idx):
            if i == 0:
                print('Running block permutations:')
            print('Block {}/{}: {}'.format(i + 1, len(to_perm_as_block_idx), feature_labels[block]))
            t1 = time()

            X_perm = X_C.copy()
            res = Parallel(n_jobs=N_JOBS, verbose=1, pre_dispatch='2*n_jobs', max_nbytes='50M') \
                (delayed(permute_feature)(clone(estimator), X_perm, y, block) for _ in range(N_PERM))

            df_feat_imp.iloc[cv_iters:, block] = res
            t2 = time()

            print('Finished in {} min \n'.format(np.round((t2 - t1) / 60., 2)))
    else:
        to_perm_columnwise = np.arange(X_C.shape[1])

    # Iterate over features that have to be permuted column wise
    for i_feature in to_perm_columnwise:
        t1 = time()

        print('{}/{}; Feature: {}'.format(i_feature + 1, X_C.shape[1], feature_labels[i_feature]))
        X_perm = X_C.copy()
        res = Parallel(n_jobs=N_JOBS, verbose=1, pre_dispatch='2*n_jobs', max_nbytes='100M') \
            (delayed(permute_feature)(clone(estimator), X_perm, y, i_feature) for _ in range(N_PERM))

        df_feat_imp.iloc[cv_iters:, i_feature] = res
        t2 = time()
        print('Finished in {} min \n'.format(np.round((t2 - t1) / 60., 2)))

    print('\n')

    dir_path = osp.join(RESULTS_DIR, "Feature_Importances")
    ensure_folder(dir_path)
    df_feat_imp.to_csv(osp.join(dir_path, trimmed_filename + "_perm_importances_brfc.csv"))


RESULTS_DIR = "/data/wbbruin/PycharmProjects/ENIGMA_OCD_ML_2020/Results"
BRFC_IDX = np.where(np.array(CLF_LABELS) == 'brfc')[0][0]
N_PERM = 500
N_JOBS = 5
N_JOBS_RF = 20

# Result(s) to perform permutations on
file_suffix = 'StratifiedKFoldByGroups_StratifiedKFoldByGroups_FS_SWN_0*_COV_Age-Sex-age_group_tesla_site_SWN_000_REG_'\
              '111_3_combined_all_cov_regressed_custom_removal_Age_Sex_age_group_tesla_site_performance_metrics.npz'

files = os.listdir(RESULTS_DIR)
results = [f for f in files if file_suffix in f]

print(results)
for r in results:
    run(r)
