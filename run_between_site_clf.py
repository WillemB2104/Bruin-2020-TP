""""
Perform between site classification benchmark for ENIGMA OCD data.

Usage:
    run_between_site_clf.py ANALYSIS_ID [options]

Arguments:
    ANALYSIS_ID             Combination of cross validation schemes used for between site classification. Can be one of
                            the following options.

                            1. Outer CV: Site-stratified fixed fold sizes,  Inner CV: Site-stratified fixed fold sizes
                            2. Outer CV: Leave One Group Out,               Inner CV: Group K-Fold
                            3. Outer CV: Site-stratified mixed fold sizes,  Inner CV: Site-stratified fixed fold sizes


Examples:
    run_between_site_clf.py 2
    run_between_site_clf.py 1 --covariates=Age,Sex --cov_sw_norm_mask=True,False
    run_between_site_clf.py 3 --fs_sw_norm --age_group=2_pediatric
    run_between_site_clf.py 2 --fs_sw_norm --covariates=Age,Sex --cov_to_regr_mask=True,True
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site  --disable_imputation
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site --query_str="(Med == 0 and Dx == 0) or (Med == 2 and Dx == 1)" --custom_suffix=filter_Med02
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site --query_str="(AO >= 18 and Dx == 1) or Dx == 0" --custom_suffix=filter_LateOnset18
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site --query_str="(Med == 1 or Med == 2) and (Dx == 1 and Sev <= 24)" --custom_suffix=Med1vsMed2_filterLowSev24_Med12 --y_label=Med
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site --query_str="(Med == 1 or Med == 2) and (Dx == 1)" --custom_suffix=Med1vsMed2_filter_Med12_Dx1 --y_label=Med
    run_between_site_clf.py 1 --covariates=Age,Sex,age_group_tesla_site --query_str="(Med == 1 or Med == 2) and (Dx == 1)" --custom_suffix=Med1vsMed2_filter_Med12_Dx1_cov_only --y_label=Med --use_covariates_only


Options:
    --fs_sw_norm
    --fs_sw_norm_mask FS_SWN_MASK
    --covariates <COVARIATES, COVARIATES>
    --cov_sw_norm_mask COV_SWN_MASK
    --cov_to_regr_mask COV_REG_MASK
    --custom_suffix CUSTOM_SUFFIX
    --age_group AGE_GROUP
    --query_str QUERY_STR
    --y_label Y_LABEL
    --disable_imputation
    --use_covariates_only
"""

from os import path as osp

from docopt import docopt
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from CustomCVs import StratifiedKFoldByGroups, StratifiedKFoldMixedSizes
from data_handling import create_data_set
from evaluation_classifier import Evaluater
from helper_functions import *

DEFAULT_SEED = 0
N_JOBS = 20
CLF_LABELS = ['svm', 'pca_svm', 'rbf_svm', 'pca_rbf_svm', 'log', 'pca_log', 'xgboost', 'brfc', 'gpc']
SAVE_DATA = '/data/wbbruin/PycharmProjects/ENIGMA_OCD_ML_2020/Results'

CUSTOM_DROP_COLUMNS = [] # ['Age', 'Sex', 'age_group_tesla_site']  # ['Age'] # Use this regress out given covariates


def run_classification(inner_cv, outer_cv, _fs_sw_norm=True, _fs_sw_norm_mask=None, _cov_sw_norm=False,
                       _cov_sw_norm_mask=None, _cov_to_regress_mask=None, save_models=True, custom_suffix=None,
                       args_dict=None, use_covariates_only=False, **kwargs):

    # Propagate optional arguments to data handling function for error handling.
    try:
        X, fs_labels, C, y, groups = create_data_set(**kwargs)
    except RuntimeError:
        raise

    # import ipdb
    # ipdb.set_trace()

    # Save arguments and parse filename
    parsed_args = args_dict.copy()
    parsed_args = {k.replace('--', ''): v for k, v in parsed_args.items()}
    parsed_args.update(kwargs)
    parsed_args['inner_cv'] = inner_cv
    parsed_args['outer_cv'] = outer_cv

    if use_covariates_only:
        X = np.zeros((X.shape[0], 0))

    # Create masks to indicate (1) which features are categorical and (2) which features should be imputed site-wise
    # or on the complete sample
    _cat_cov_mask, _cov_sw_norm_mask, _cov_to_regress_mask, _fs_sw_norm_mask, cols_to_imp_on_tot, cols_to_imp_sw, \
    cols_to_norm_and_imp_on_tot, cols_to_norm_and_imp_sw, covariates = create_normalization_masks(X, _fs_sw_norm,
                                                                                                  _fs_sw_norm_mask, C,
                                                                                                  _cov_sw_norm,
                                                                                                  _cov_sw_norm_mask,
                                                                                                  _cov_to_regress_mask,
                                                                                                  kwargs)

    # Merge FreeSurfer data (X) with covariates (C); save labels accordingly
    if use_covariates_only:
        X_C = C
        X_C_labels = covariates
    else:
        X_C = np.c_[X, C]
        X_C_labels = np.append(fs_labels, covariates) if C.size > 0 else fs_labels  # prevent appending None's


    outer_cv_label, inner_cv_label = extract_cv_label(outer_cv), extract_cv_label(inner_cv)
    outer_cv_state, inner_cv_state = get_random_state(outer_cv), get_random_state(inner_cv)

    outer_cv_splits = outer_cv.get_n_splits(X, y, groups)
    inner_cv_splits = inner_cv.n_splits

    evaluater = Evaluater()
    metrics_labels = evaluater.evaluate_labels()

    metrics = np.zeros((len(CLF_LABELS), len(metrics_labels), outer_cv_splits))
    roc_curves = np.zeros((len(CLF_LABELS)), dtype=object)
    clf_models = np.zeros((len(CLF_LABELS), outer_cv_splits), dtype=object)

    # initialized to -1: if a subject wasn't chosen for the iteration it will remain -1
    predictions = np.ones((len(CLF_LABELS), y.size, outer_cv_splits)) * -1
    scores = np.ones((len(CLF_LABELS), y.size, outer_cv_splits)) * -1

    filename_prefix = create_filename(C.size, _cov_sw_norm_mask, _cov_to_regress_mask, _fs_sw_norm_mask, covariates,
                                      inner_cv_label, outer_cv_label, custom_suffix, parsed_args)
    if np.any(_cov_to_regress_mask) and (len(set(X_C_labels) & set(CUSTOM_DROP_COLUMNS)) > 0):
        filename_prefix += 'custom_removal_' + '_'.join(CUSTOM_DROP_COLUMNS) + '_'

    skipped_iterations = []

    for id_iter_cv, (train_id, test_id) in enumerate(outer_cv.split(X_C, y, groups)):

        # Create copy of masks and labels as these might be mutated later
        tmp_covariates = covariates
        tmp_cat_cov_mask = _cat_cov_mask
        tmp_cov_to_regress_mask = _cov_to_regress_mask
        tmp_X_C_labels = X_C_labels

        print("Iteration: {}".format(id_iter_cv + 1))
        X_C_train, X_C_test = X_C[train_id], X_C[test_id]
        y_train, y_test = y[train_id], y[test_id]
        groups_train, groups_test = groups[train_id], groups[test_id]

        if len(np.unique(y_train)) == 1:
            print("Only one class present in y_train, skipping iteration {}".format(id_iter_cv + 1))
            skipped_iterations.append(id_iter_cv)
            continue
        elif len(np.unique(y_test)) == 1:
            print("Only one class present in y_test, cannot compute metrics for this iteration")
            skipped_iterations.append(id_iter_cv)
            continue

        print("nans in train: {}, nans in test: {}\n"
              "X_C_train shape: {}, X_C_test shape: {}".format(np.isnan(X_C_train).sum(), np.isnan(X_C_test).sum(),
                                                               X_C_train.shape, X_C_test.shape))

        # Check if there are any features that have to be normalized site-wise
        if np.any(np.append(_fs_sw_norm_mask, _cov_sw_norm_mask)):
            # Site-wise normalization parameters should be estimated separately for training and test with LOGO-CV
            logo_cv = True if "LeaveOneGroupOut" in (inner_cv_label, outer_cv_label) else False
            print("# Perform site-wise normalization for columns in given masks")
            X_C_train, X_C_test = site_wise_normalization(X, C, _cov_sw_norm_mask, _fs_sw_norm_mask,
                                                          cols_to_imp_sw, cols_to_norm_and_imp_sw,
                                                          tmp_covariates, fs_labels, groups, train_id, test_id, logo_cv)

        print("nans in train: {}, nans in test: {}\n"
              "X_C_train shape: {}, X_C_test shape: {}".format(np.isnan(X_C_train).sum(), np.isnan(X_C_test).sum(),
                                                               X_C_train.shape, X_C_test.shape))

        # Check if there are any features that have to be normalized over total sample
        if np.any(~np.append(_fs_sw_norm_mask, _cov_sw_norm_mask)):
            print("# Perform normalization over total sample for columns in given masks")
            X_C_train, X_C_test = total_normalization(X_C_train, X_C_test, cols_to_imp_on_tot,
                                                      cols_to_norm_and_imp_on_tot)

        print("nans in train: {}, nans in test: {}\n"
              "X_C_train shape: {}, X_C_test shape: {}".format(np.isnan(X_C_train).sum(), np.isnan(X_C_test).sum(),
                                                               X_C_train.shape, X_C_test.shape))

        if np.any(tmp_cat_cov_mask):
            print("# Encode categorical covariates into dummy variables")
            X_C_test, X_C_train, tmp_cat_cov_mask, tmp_cov_to_regress_mask, tmp_covariates, tmp_X_C_labels = \
                encode_cat_2_dummy(X_C_test, X_C_train, tmp_cat_cov_mask, tmp_cov_to_regress_mask, tmp_covariates,
                                   tmp_X_C_labels, test_id, train_id)

        # Save info which sample used for pred, and what labels used per CV iteration
        print("nans in train: {}, nans in test: {}\n"
              "X_C_train shape: {}, X_C_test shape: {}".format(np.isnan(X_C_train).sum(), np.isnan(X_C_test).sum(),
                                                               X_C_train.shape, X_C_test.shape))

        # import ipdb
        # ipdb.set_trace()

        if np.any(tmp_cov_to_regress_mask):
            print("# Regress out covariates")
            X_C_test, X_C_train = regress_out_cov(X, tmp_X_C_labels, X_C_test, X_C_train, tmp_cov_to_regress_mask,
                                                  tmp_covariates, train_id)

            dummy_labels = [l for l in tmp_X_C_labels if any(c in l for c in CUSTOM_DROP_COLUMNS)]

            # Custom column removals
            if len(set(tmp_X_C_labels) & set(dummy_labels)) > 0:
                # Check if CUSTOM_DROP_COLUMN is categorical. if so, update this one accordingly
                dummy_labels = [l for l in tmp_X_C_labels if any(c in l for c in CUSTOM_DROP_COLUMNS)]
                mask = [tmp not in dummy_labels for tmp in tmp_X_C_labels]
                print('# Dropping specified columns after regression: {}'.format(CUSTOM_DROP_COLUMNS))
                X_C_train = X_C_train[:, mask]
                X_C_test = X_C_test[:, mask]
                print("nans in train: {}, nans in test: {}\n"
                      "X_C_train shape: {}, X_C_test shape: {}".format(np.isnan(X_C_train).sum(),
                                                                       np.isnan(X_C_test).sum(),
                                                                       X_C_train.shape, X_C_test.shape))

        clfs, clfs_labels = get_models_to_check()

        n_jobs = N_JOBS
        clf_index = 0
        t1 = time()

        # Evaluate models with nested grid-search first (svm, pca_svm, rbf_svm, pca_rbf_svm, log, pca_log, xgboost)
        for clf, clf_label in zip(clfs, clfs_labels):

            print('Performing grid search for: {} \n'.format(clf_label))
            t2 = time()
            clf, results_df = inner_loop_iteration(clf, clf_label, X_C_train, y_train, groups_train, inner_cv, n_jobs)
            t3 = time()
            print('Finished grid search in {:.2f} minutes \n'.format((t3 - t2) / 60.))

            t4 = time()
            clf.fit(X_C_train, y_train)
            t5 = time()
            print('Fitted {} in {:.2f} minutes \n'.format(clf_label, (t5 - t4) / 60.))

            y_pred = clf.predict(X_C_test)
            y_score = clf.predict_proba(X_C_test)[:, 1]

            metrics[clf_index, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                              y_score=y_score)
            evaluater.print_evaluation()
            predictions[clf_index, test_id, id_iter_cv] = y_pred
            scores[clf_index, test_id, id_iter_cv] = y_score
            fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
            roc_curves[clf_index] = np.append(roc_curves[clf_index], [fpr, tpr, threshold])
            if save_models:
                clf_models[clf_index, id_iter_cv] = clf
            print('\n')
            clf_index += 1

        print("Running bRFC ...")
        brfc, y_pred, y_score = brfc_classification(X_C_train, y_train, X_C_test, n_jobs)
        metrics[-2, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y_test, y_pred=y_pred, y_score=y_score)
        print('Finished evaluation for bRFC, results: \n')
        evaluater.print_evaluation()
        predictions[-2, test_id, id_iter_cv] = y_pred
        scores[-2, test_id, id_iter_cv] = y_score
        fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
        roc_curves[-2] = np.append(roc_curves[-1], [fpr, tpr, threshold])
        if save_models:
            clf_models[-2, id_iter_cv] = brfc


        print("Running GPC ...")
        GPC_clf, y_pred, y_score = gpc_classification(X_C_train, y_train, X_C_test, y_test)
        metrics[-1, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y[test_id], y_pred=y_pred,
                                                                   y_score=y_score)

        print('Finished evaluation for GPC, results: \n')
        evaluater.print_evaluation()
        predictions[-1, test_id, id_iter_cv] = y_pred
        scores[-1, test_id, id_iter_cv] = y_score
        fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
        roc_curves[-1] = np.append(roc_curves[-1], [fpr, tpr, threshold])
        if save_models:
            clf_models[-1, id_iter_cv] = GPC_clf

        t4 = time()
        print("Finished fold {}/{} in {:.2f} minutes".format(id_iter_cv + 1, outer_cv_splits, (t4 - t1) / 60.))

    print("Saving data")

    if len(skipped_iterations) > 0:
        it_mask = np.ones(outer_cv_splits, dtype=bool)
        it_mask[skipped_iterations] = False
        metrics = metrics[:, :, it_mask]
        predictions = predictions[:, :, it_mask]
        if save_models:
            clf_models = clf_models[:, it_mask]

        np.save(osp.join(SAVE_DATA, filename_prefix + "skipped_iterations.npy"), skipped_iterations)

    np.savez_compressed(osp.join(SAVE_DATA, filename_prefix + "performance_metrics.npz"), metrics=metrics,
                        metrics_labels=metrics_labels)

    np.save(osp.join(SAVE_DATA, filename_prefix + "arguments.npy"), parsed_args)
    np.save(osp.join(SAVE_DATA, filename_prefix + "predictions.npy"), predictions)
    np.save(osp.join(SAVE_DATA, filename_prefix + "roc_curves.npy"), roc_curves)
    np.save(osp.join(SAVE_DATA, filename_prefix + "scores.npy"), scores)

    if save_models:
        np.save(osp.join(SAVE_DATA, filename_prefix + "clf_models.npy"), clf_models)


if __name__ == '__main__':

    args = docopt(__doc__)

    implemented_analyses = ['1', '2', '3']
    analysis_ID = args['ANALYSIS_ID']

    age_group = args['--age_group']
    fs_sw_norm = args['--fs_sw_norm']
    disable_imputation = args['--disable_imputation']

    covariates = str_to_list(args['--covariates'], str)

    fs_sw_norm_mask = str_to_list(args['--fs_sw_norm_mask'], bool)
    cov_sw_norm_mask = str_to_list(args['--cov_sw_norm_mask'], bool)
    cov_to_regr_mask = str_to_list(args['--cov_to_regr_mask'], bool)

    query_str = args['--query_str']
    custom_suffix = args['--custom_suffix']
    y_label = args['--y_label']
    only_covariates = args['--use_covariates_only']

    print(args)

    if analysis_ID not in implemented_analyses:
        raise RuntimeError(
            'Not recognized analysis ID {}. Currently only analysis IDs "1", "2" and "3" are implemented'
                .format(analysis_ID))
    elif analysis_ID == '1':
        print('1. Outer CV: Site-stratified fixed fold sizes, Inner CV: Site-stratified fixed fold sizes')
        outer_cv = StratifiedKFoldByGroups(n_splits=10, random_state=DEFAULT_SEED, shuffle=True)
        inner_cv = StratifiedKFoldByGroups(n_splits=5, random_state=DEFAULT_SEED, shuffle=True)
    elif analysis_ID == '2':
        print('2. Outer CV: Leave One Group Out, Inner CV: Group K-Fold')
        outer_cv = LeaveOneGroupOut()
        inner_cv = GroupKFold(n_splits=5)
    elif analysis_ID == '3':
        print('3. Outer CV: Site-stratified mixed fold sizes, Inner CV: Site-stratified fixed fold sizes')
        outer_cv = StratifiedKFoldMixedSizes(random_state=DEFAULT_SEED)
        inner_cv = StratifiedKFoldByGroups(n_splits=5, random_state=DEFAULT_SEED, shuffle=True)

    if age_group:
        run_classification(inner_cv=inner_cv, outer_cv=outer_cv, verbose=True, covariates=covariates,
                           min_counts_per_site="auto", query_str=query_str, custom_suffix=custom_suffix,
                           _fs_sw_norm_mask=fs_sw_norm_mask, _fs_sw_norm=fs_sw_norm,
                           use_covariates_only=only_covariates, _cov_to_regress_mask=cov_to_regr_mask,
                           y_label=y_label, complete=disable_imputation, _cov_sw_norm_mask=cov_sw_norm_mask,
                           age_group=age_group, args_dict=args)
    else:
        for sample in [None, '1_adult', '2_pediatric']:
            run_classification(inner_cv=inner_cv, outer_cv=outer_cv, verbose=True, covariates=covariates,
                               min_counts_per_site="auto", query_str=query_str, custom_suffix=custom_suffix,
                               _fs_sw_norm_mask=fs_sw_norm_mask, _fs_sw_norm=fs_sw_norm,
                               use_covariates_only=only_covariates, _cov_to_regress_mask=cov_to_regr_mask,
                               y_label=y_label, complete=disable_imputation, _cov_sw_norm_mask=cov_sw_norm_mask,
                               age_group=sample, args_dict=args)
