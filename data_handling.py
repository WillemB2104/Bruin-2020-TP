import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join("/data/wbbruin/Desktop/ENIGMA_ML_BENCHMARK", "Data",
                        "ENIGMA_OCD_26-01-2019.csv")  # Set this path accordingly


def ensure_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df


def find_min_counts_per_site(counts):
    threshold = 0
    complete = False
    while not complete:
        n_selected = sum(counts > threshold)
        min_groups = min(counts[counts > threshold])
        if not min_groups < n_selected:
            complete = True
        else:
            threshold += 1
    return threshold


def create_data_set(complete=False, completeness_threshold=0.9, age_group=None, covariates=None, query_str=None,
                    min_counts_per_site=None, csv_path=CSV_PATH, y_label=None, verbose=False):
    """
    Create ENIGMA data set for classification with given constraints.

    :param complete : boolean, default=False
        Return data without missing values by dropping all rows that contain any NA entry.
    :param completeness_threshold : float, default=0.9
        Specifies threshold for the minimum proportion of complete entries that each row should have (will only be in
        effect if complete = False). Threshold should be in the range between 0. (no threshold is used) and 1. (drop
        rows that contain any NA entry).
    :param age_group : str, default=None
        Limit data set to specific age group (either "1_adult", " 2_pediatric" or None to include all participants).
    :param covariates : str or list, default=None
        Covariates that will be returned from the data set.
    :param query_str: str, default=None
        Query used to filter ENIGMA data frame for specific sample (e.g. 'Sex ==  1 & Med == 1')
    :param min_counts_per_site: int or str, default=None
        Specifies threshold of the minimum sample size per site (either fixed by passing an int, found automatically by
        using "auto" or None by default).
    :param csv_path : str, default=CSV_PATH
        Path to ENIGMA csv file. Default is set in CSV_PATH global.
    :param y_label : str, default=None,
	    Label for custom variable that will be returned as y. Set to "Dx" (OCD diagnosis) by default.
    :param verbose : boolean, default False
        Enable prints for detailed logging information.
    :return: X, selected_fs_features, C, y, groups
    """

    df = load_csv(csv_path)

    print('Loading ENIGMA dataset: Complete = {}, Min_Threshold = {}, Age_Group = {}, Covariates = {}, '
          'Min_Counts_per_Site = {}, y_label \n'.format(complete, completeness_threshold, age_group, covariates,
                                                        min_counts_per_site, y_label))

    # print('Index of 1st FreeSurfer feature: {} \n'.format(np.where(df.columns.values == 'subcort_ICV')[0]))
    first_fs_index = np.where(df.columns.values == 'subcort_ICV')[0]
    subcort_idx = np.where(df.columns.values == "LLatVent")[0]
    cort_surf_idx = np.where(df.columns.values == "LSurfArea")[0]
    cort_thick_idx = np.where(df.columns.values == "LThickness")[0]

    # Create unique group (site) labels if not made already.
    if "age_group_tesla_site" not in df.columns:
        df['age_group_tesla_site'] = df["age_group"] + df["tesla"] + df["site"]

    # Standard approach will use all available FS features (excluding final column as this contains site IDs).
    fs_labels = df.columns.values[np.arange(first_fs_index, len(df.columns) - 1)]
    cov_labels = np.append(df.columns.values[np.arange(0, first_fs_index)], ["age_group_tesla_site", "Dx"])
    age_group_labels = np.unique(df.age_group)

    y_label = y_label if y_label else "Dx"

    if age_group:
        if age_group not in age_group_labels:
            raise RuntimeError('Age group {} not an option. Options are {}'
                               .format(age_group, age_group_labels))
        else:
            df = df[df.age_group == age_group]

    if covariates:
        if isinstance(covariates, str):
            covariates = [covariates]
        if not isinstance(covariates, list):
            raise RuntimeError('TypeError: covariates must be str or list, {} not accepted \n'.format(type(covariates)))
        if not set(covariates).issubset(set(cov_labels)):
            raise RuntimeError('Warning! Unknown covariates specified: {} \n'
                               'Only the following options are allowed: {} \n'.format(covariates, cov_labels))
        else:
            all_selected_features = np.append(fs_labels, covariates)
    else:
        all_selected_features = fs_labels

    if verbose:
        print('Features to extract ({}/{}): \n {} \n'.format(len(all_selected_features), len(fs_labels) +
                                                             len(cov_labels), all_selected_features))

    if complete:
        # Remove subjects that miss ANY feature
        tmp_df = df.loc[~df.loc[:, np.append(y_label, all_selected_features)].T.isnull().any()].copy()
    else:
        # Remove subjects that miss target label
        n_ = df.shape[0]
        tmp_df = df.loc[~df.loc[:, [y_label]].T.isnull().any()].copy()
        if verbose:
            print('Subjects excluded with missing target label ({}) : {} out of {} \n \n'.format(y_label,
                                                                                                 n_ - tmp_df.shape[0],
                                                                                                 n_))

        # Remove subjects which miss more features than given threshold
        nans_per_subject = tmp_df.loc[:, all_selected_features].isnull().T.sum()
        completeness_per_subject = (float(tmp_df.shape[1]) - nans_per_subject) / tmp_df.shape[1]
        tmp_df = tmp_df.loc[completeness_per_subject >= completeness_threshold]
        n_dropped = sum(completeness_per_subject < completeness_threshold)

        if verbose:
            print('Subjects excluded with < {}% feature completeness: {} \n \n'.format(completeness_threshold * 100,
                                                                                       n_dropped))

    if query_str:
        try:
            n_org = tmp_df.shape[0]
            tmp_df = tmp_df.query(query_str)
            n_fil = tmp_df.shape[0]
            if verbose:
                print('Query used: {}, subjects excluded = {} (out of {}) \n \n'.format(query_str, n_org-n_fil, n_org))
        except RuntimeError:
            raise

    groups = tmp_df.loc[:, "age_group_tesla_site"].values
    sites, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    tmp_df.loc[:, "age_group_tesla_site"] = inverse

    X = tmp_df.loc[:, fs_labels].values
    y = tmp_df.loc[:, y_label].values.astype(int)
    C = tmp_df.loc[:, covariates].values if covariates else np.array([]).reshape((X.shape[0], 0))

    # (Hard)code y labels for now
    if y_label == "Med":
        y = y - 1
    elif y_label == "Sev":
        y = np.array(y > 24, dtype=int)
    elif y_label == "Dur":
        y = np.array(y > 7, dtype=int)
    elif y_label == "AO":
        y = np.array(y >= 18, dtype=int)

    threshold = 0

    if min_counts_per_site:
        if min_counts_per_site == 'auto':
            threshold = find_min_counts_per_site(counts)
        elif isinstance(min_counts_per_site, int):
            threshold = min_counts_per_site
        else:
            raise RuntimeError('Invalid option for min_counts_per_site: {}, only integers and "auto" are allowed'.
                               format(min_counts_per_site))

    included_sites = sites[counts > threshold]
    mask = np.isin(groups, included_sites)

    if verbose and sum(mask) is not len(mask):
        print('Minimum subjects per site threshold: {},\n Sites excluded: {}, Subjects excluded: {}\n'
              .format(threshold, np.unique(groups[~mask]), len(mask) - sum(mask)))

    X, C, y, groups = X[mask], C[mask], y[mask], groups[mask]

    print('Finished loading data set: {} samples, {} FS features, {} covariates \n \n '.format(X.shape[0], X.shape[1],
                                                                                               C.shape[1] if covariates
                                                                                               else 0))

    return X, fs_labels, C, y, groups
