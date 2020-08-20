from os import path as osp

from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from data_handling import create_data_set, ensure_folder
from evaluation_classifier import Evaluater
from helper_functions import *
from helper_functions import _handle_zeros_in_scale
from run_between_site_clf import CLF_LABELS

DEFAULT_SEED = 0
N_JOBS = 20
SAVE_DATA = '/data/wbbruin/PycharmProjects/ENIGMA_OCD_ML_2020/Results/Single_Site_Results_5CV'


def run(impute='Median', save_models=False, save_gridsearch=False):
    covariates = ['Age', 'Sex']
    X, fs_labels, C, y_total, group_total = create_data_set(complete=False, completeness_threshold=0.9,
                                                            covariates=covariates, min_counts_per_site=None,
                                                            y_label="Dx", verbose=True)

    X_total = np.c_[X, C]
    columns_to_choose = np.append(fs_labels, covariates) if C.size > 0 else fs_labels  # prevent appending None's

    unique_sites, counts = np.unique(group_total, return_counts=True)

    outer_cv_splits = 5
    inner_cv_splits = 5
    n_repeats = 10

    for site_id in unique_sites:

        dir_path = osp.join(SAVE_DATA, "Site_" + str(site_id))

        # Check if dir_path exists, if so, skip it!
        if osp.exists(dir_path):
            print("Output already exists for {}, skipping!\n".format(dir_path))
            continue

        print('Running single site classification for site {} \n'.format(site_id))
        mask = group_total == site_id
        X, y, groups = X_total[mask], y_total[mask], group_total[mask]

        inner_cv = StratifiedKFold(shuffle=True, random_state=DEFAULT_SEED, n_splits=inner_cv_splits)
        outer_cv = RepeatedStratifiedKFold(random_state=DEFAULT_SEED, n_repeats=n_repeats, n_splits=outer_cv_splits)

        outer_cv_label = extract_cv_label(outer_cv)
        inner_cv_label = extract_cv_label(inner_cv)

        evaluater = Evaluater()

        fold_split_sizes = np.zeros((outer_cv_splits * n_repeats, 2))
        metrics_labels = evaluater.evaluate_labels()
        metrics = np.zeros((len(CLF_LABELS), len(metrics_labels), outer_cv_splits * n_repeats))
        roc_curves = np.zeros((len(CLF_LABELS)), dtype=object)
        clf_models = np.zeros((len(CLF_LABELS), outer_cv_splits * n_repeats), dtype=object)

        # initialized to -1: if a subject wasn't chosen for the iteration it will remain -1
        predictions = np.ones((len(CLF_LABELS), y.size, outer_cv_splits * n_repeats)) * -1

        print('Inner CV = {} , n_splits = {} \n'.format(inner_cv_label, inner_cv.get_n_splits()))
        print('Outer CV = {} , n_splits = {} \n'.format(outer_cv_label, outer_cv.get_n_splits()))
        print('Imputation = {} \n'.format(impute))

        try:
            for id_iter_cv, (train_id, test_id) in enumerate(outer_cv.split(X, y)):

                print("Iteration: {}".format(id_iter_cv + 1))

                X_train, X_test = X[train_id], X[test_id]
                y_train, y_test = y[train_id], y[test_id]

                fold_split_sizes[id_iter_cv, 0] = np.size(train_id)
                fold_split_sizes[id_iter_cv, 1] = np.size(test_id)

                clfs, clfs_labels = get_models_to_check()
                n_jobs = N_JOBS
                t1 = time()
                clf_random_state = 1
                clf_index = 0

                # 1. Estimate median (using training data only!)
                tmp_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
                tmp_imputer.fit(X_train)
                center_ = tmp_imputer.statistics_

                # 2. Estimate interquartile range (using training data only!)
                q_min, q_max = 25.0, 75.0
                q = np.nanpercentile(X_train, [q_min, q_max], axis=0)
                scale_ = (q[1] - q[0])
                scale_ = _handle_zeros_in_scale(scale_, copy=False)

                # 3. Impute median
                X_train = tmp_imputer.transform(X_train)
                X_test = tmp_imputer.transform(X_test)

                # 4. Robust scale everything except for Sex (final feature column)!
                X_train[:, :-1] -= center_[:-1]
                X_train[:, :-1] /= scale_[:-1]
                X_test[:, :-1] -= center_[:-1]
                X_test[:, :-1] /= scale_[:-1]

                # Evaluate models with nested grid-search first
                for clf, clf_label in zip(clfs, clfs_labels):

                    # import ipdb
                    # ipdb.set_trace()

                    t2 = time()
                    print('Performing grid search for: {} \n'.format(clf_label))
                    clf, results_df = inner_loop_iteration(clf, clf_label, X_train, y_train, None, inner_cv,
                                                           n_jobs)
                    t3 = time()
                    print('Finished grid search in {:.2f} minutes \n'.format((t3 - t2) / 60.))

                    clf_state = get_random_state(clf)

                    t5 = time()
                    clf.fit(X_train, y_train)
                    t6 = time()
                    print('Fitted {} in {:.2f} minutes \n'.format(clf_label, (t6 - t5) / 60.))

                    y_pred = clf.predict(X_test)
                    y_score = clf.predict_proba(X_test)[:, 1]
                    metrics[clf_index, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                                      y_score=y_score)
                    evaluater.print_evaluation()
                    predictions[clf_index, test_id, id_iter_cv] = y_pred
                    fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)

                    roc_curves[clf_index] = np.append(roc_curves[clf_index], [fpr, tpr, threshold])

                    if save_models:
                        clf_models[clf_index, id_iter_cv] = clf

                    print('\n')
                    clf_index += 1

                print("Running bRFC ...")
                brfc, y_pred, y_score = brfc_classification(X_train, y_train, X_test, n_jobs)
                metrics[-2, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y_test, y_pred=y_pred,
                                                                           y_score=y_score)
                print('Finished evaluation for bRFC, results: \n')
                predictions[-2, test_id, id_iter_cv] = y_pred
                evaluater.print_evaluation()
                fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
                roc_curves[-2] = np.append(roc_curves[-1], [fpr, tpr, threshold])
                if save_models:
                    clf_models[-2, id_iter_cv] = brfc

                print("Running GPC ...")
                GPC_clf, y_pred, y_score = gpc_classification(X_train, y_train, X_test, y_test)
                metrics[-1, :, id_iter_cv] = evaluater.evaluate_prediction(y_true=y[test_id], y_pred=y_pred,
                                                                           y_score=y_score)
                print('Finished evaluation for GPC, results: \n')
                predictions[-1, test_id, id_iter_cv] = y_pred
                evaluater.print_evaluation()
                fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=y_score)
                roc_curves[-1] = np.append(roc_curves[-1], [fpr, tpr, threshold])
                if save_models:
                    clf_models[-1, id_iter_cv] = GPC_clf

                print('\n')
                t4 = time()
                print("Finished fold {}/{} in {:.2f} minutes".format(id_iter_cv + 1, outer_cv_splits, (t4 - t1) / 60.))

            print("\n\n DONE \n\n")
            print("Test fold size mean = {}, variance = {} \n".
                  format(np.nanmean(fold_split_sizes), np.nanvar(fold_split_sizes)))

            print("Saving data")

            ensure_folder(dir_path)

            prefix = "Site_" + str(site_id) + "_" + outer_cv_label + "_" + inner_cv_label + "_"
            np.savez_compressed(osp.join(dir_path, prefix + "performance_metrics.npz"), metrics=metrics,
                                metrics_labels=metrics_labels)
            np.save(osp.join(dir_path, prefix + "predictions.npy"), predictions)
            np.save(osp.join(dir_path, prefix + "roc_curves.npy"), roc_curves)

            if save_models:
                np.save(osp.join(dir_path, prefix + "clf_models.npy"), clf_models)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    ensure_folder(SAVE_DATA)
    run(impute='Median', save_models=True, save_gridsearch=False)
