# coding: utf-8

from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_random_state, check_array


# Ignore these two FutureWarnings
warnings.filterwarnings("ignore", message="Pass shuffle=True, random_state=0 as keyword args", )
warnings.filterwarnings("ignore", message="Pass groups")



def round_off_matrix(fn):
    rows, columns = fn.shape

    # Temporary matrices with same dimensions as fn.
    tmp_res, tmp_diff, tmp_idx = np.zeros_like(fn, dtype=int), np.zeros_like(fn, dtype=float), \
                                 np.zeros_like(fn, dtype=object)

    # Calculate the expected sum across columns
    rowsSum = np.rint(fn.sum(axis=1)).astype(int)

    # Populate temp matrices.
    for r in range(rows):
        for c in range(columns):
            tmp_res[r, c] = np.floor(fn[r, c])  # Lower bound
            tmp_diff[r, c] = fn[r, c] - np.floor(fn[r, c])  # Roundoff error
            tmp_idx[r, c] = [r, c]  # Original index

    # Count the lower bound rounded sum across columns
    lowerSum = np.rint(tmp_res.sum(axis=1)).astype(int)

    # Calculate difference between expected sum versus rounded sum across columns
    difference = rowsSum - lowerSum

    # Sort each row in matrices on roundoff error
    sorted_indices = [row[::-1] for row in np.argsort(tmp_diff, axis=1)]
    tmp_res = tmp_res[np.arange(rows)[:, np.newaxis], sorted_indices]
    tmp_diff = tmp_diff[np.arange(rows)[:, np.newaxis], sorted_indices]
    tmp_idx = tmp_idx[np.arange(rows)[:, np.newaxis], sorted_indices]

    # Increment the lower bound values where roundoff error is the greatest
    for r in range(rows):
        for c in range(difference[r]):
            tmp_res[r, c] = tmp_res[r, c] + 1

    # Sort results back to original indices
    sorted_res = np.zeros_like(fn, dtype=int)
    for r in range(rows):
        for c in range(columns):
            original_index = tmp_idx[r, c]
            result = tmp_res[r, c]
            sorted_res[original_index[0], original_index[1]] = result

    # Sanity check: verify whether sum across columns is the same for input and output matrix
    if not np.all(np.rint(fn.sum(axis=1)) == sorted_res.sum(axis=1)):
        print("Warning! Original row-wise sum = {}, Roundoff row-wise sum = {}, difference = {}".format(
            np.rint(fn.sum(axis=1)), sorted_res.sum(axis=1), np.rint(fn.sum(axis=1)) - sorted_res.sum(axis=1)))

    return sorted_res


class StratifiedKFoldMixedSizes(BaseCrossValidator):
    """Stratified KFold cross validator for folds of mixed sizes.

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers. Fold sizes
    are set to match respective counts of unique groups by default. Custom
    fold sizes can be set with the custom_fold_sizes.

    """

    def __init__(self, random_state=None, custom_fold_sizes=None):
        self.random_state = random_state
        self.custom_fold_sizes = custom_fold_sizes

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups, counts = np.unique(groups, return_counts=True)

        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups)

        # Calculate test size for each split based on group label frequency
        test_sizes = counts / float(counts.sum())

        # Fold sizes are set to match counts of unique groups by default
        custom_test_sizes = test_sizes

        if self.custom_fold_sizes is not None:
            fold_sizes = check_array(self.custom_fold_sizes, copy=True, ensure_2d=False, dtype=None)
            if len(fold_sizes) <= 1:
                raise ValueError(
                    "The custom_fold_sizes parameter contains fewer than 2 folds "
                    "(%s). StratifiedKFoldMixedSizes expects at least 2." % fold_sizes)
            custom_test_sizes = fold_sizes / float(fold_sizes.sum())

        n_sizes = np.reshape(test_sizes, (len(test_sizes), 1)) * \
                  np.reshape(custom_test_sizes, (1, len(custom_test_sizes))) * \
                  len(groups)

        n_sizes_rounded = round_off_matrix(n_sizes)

        n_folds = int(sum(n_sizes_rounded.sum(axis=0) > 0))

        if np.all(n_folds > counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class.".format(n_folds))

        if n_folds > min(counts):
            warnings.warn(("The least populated group has only {}"
                           " members, which is too few. The minimum"
                           " number of members in any group cannot"
                           " be less than n_splits={}".format(min(counts), n_folds)), Warning)

        # Shuffle available indices
        available_ids = pd.DataFrame(groups, columns=['groups'])
        available_ids = available_ids.sample(frac=1, random_state=check_random_state(self.random_state))
        available_ids = available_ids.sort_values('groups')

        raveled_indices = n_sizes_rounded.ravel()
        stacked_indices = [sum(raveled_indices[0:i + 1]) for i in range(len(raveled_indices))]

        splits = np.split(np.array(available_ids.index, dtype=int), stacked_indices[:-1])
        group_indices_per_split = np.reshape(splits, n_sizes_rounded.shape).T

        for i_split, split in enumerate(group_indices_per_split):
            test_indices = np.concatenate(split)
            if len(test_indices) == 0:
                continue
            tmp = np.zeros(len(groups), dtype=bool)
            tmp[test_indices] = 1
            yield tmp

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object, optional
            Always ignored, exists for compatibility.

        y : object, optional
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups, counts = np.unique(groups, return_counts=True)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups)

        test_sizes = counts / float(counts.sum())
        custom_test_sizes = test_sizes

        if self.custom_fold_sizes is not None:
            fold_sizes = check_array(self.custom_fold_sizes, copy=True, ensure_2d=False, dtype=None)
            if len(fold_sizes) <= 1:
                raise ValueError(
                    "The custom_fold_sizes parameter contains fewer than 2 folds "
                    "(%s). StratifiedKFoldMixedSizes expects at least 2." % fold_sizes)
            custom_test_sizes = fold_sizes / float(fold_sizes.sum())

        n_sizes = np.reshape(test_sizes, (len(test_sizes), 1)) * \
                  np.reshape(custom_test_sizes, (1, len(custom_test_sizes))) * \
                  len(groups)

        n_sizes_rounded = round_off_matrix(n_sizes)

        return int(sum(n_sizes_rounded.sum(axis=0) > 0))


class KFoldMixedSizes(BaseCrossValidator):
    """KFold Mixed Sizes cross validator build on Leave One Group Out validator

    Provides train/test indices to split data according to random permutations
    of third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups)
        groups = check_random_state(self.random_state).permutation(groups)
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object, optional
            Always ignored, exists for compatibility.

        y : object, optional
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        return len(np.unique(groups))


class StratifiedKFoldByGroups(_BaseKFold):
    """Stratified K-Fold by Groups cross-validator

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle each stratification of the data before splitting
        into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(StratifiedKFoldByGroups, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y, groups):
        rng = self.random_state
        groups = np.asarray(groups)
        n_samples = groups.shape[0]
        unique_groups, groups_inversed = np.unique(groups, return_inverse=True)
        groups_counts = np.bincount(groups_inversed)
        min_groups = np.min(groups_counts)
        if np.all(self.n_splits > groups_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of members in any class cannot"
                           " be less than n_splits=%d."
                           % (min_groups, self.n_splits)), Warning)

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        # NOTE: Passing the data corresponding to ith class say X[y==class_i]
        # will break when the data is not 100% stratifiable for all classes.
        # So we pass np.zeroes(max(c, n_splits)) as data to the KFold
        per_cls_cvs = [
            KFold(self.n_splits, shuffle=self.shuffle,
                  random_state=rng).split(np.zeros(max(count, self.n_splits)))
            for count in groups_counts]

        test_folds = np.zeros(n_samples, dtype=np.int)
        for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_groups, per_cls_splits):
                cls_test_folds = test_folds[groups == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[groups == cls] = cls_test_folds

        return test_folds

    def _iter_test_masks(self, X, y, groups):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(StratifiedKFoldByGroups, self).split(X, y, groups)
