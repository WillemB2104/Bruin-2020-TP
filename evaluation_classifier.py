from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import numpy as np
import inspect
from collections import OrderedDict


def balanced_accuracy(y_true, y_pred):
    return 0.5 * (specificity(y_true, y_pred) + recall_score(y_true, y_pred))


def ppv(y_true, y_pred):
    # noinspection PyTypeChecker
    ppv_score = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1, dtype=np.float)
    if np.isnan(ppv_score):
        return 0
    return ppv_score


def npv(y_true, y_pred):
    # noinspection PyTypeChecker
    npv_score = np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_pred == 0, dtype=np.float)
    if np.isnan(npv_score):
        return 0
    return npv_score


def specificity(y_true, y_pred):
    # noinspection PyTypeChecker
    return np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0, dtype=np.float)


def return_prediction_first(y_pred):
    return y_pred[0]


def return_prediction_second(y_pred):
    return y_pred[1]


def return_true_first(y_true):
    return y_true[1]


def return_true_second(y_true):
    return y_true[1]


class Evaluater(object):

    def __init__(self, leave_one_out_case=False):
        self.loo = leave_one_out_case
        self.evaluations = self.set_evaluations()
        self.results = OrderedDict()
        self.evaluation_string = ''

    def evaluate(self, **kwargs):
        for eval_label, eval_fun in self.evaluations.items():
            args_to_use = set([param for param in inspect.signature(eval_fun).parameters]) & set(kwargs.keys())
            args_to_use = {key: kwargs[key] for key in args_to_use}
            self.results[eval_label] = eval_fun(**args_to_use)

    def evaluate_prediction(self, **kwargs):
        self.evaluate(**kwargs)
        return list(self.results.values())

    def evaluate_labels(self):
        return list(self.evaluations.keys())

    def print_evaluation(self):
        if not self.results:
            raise RuntimeError('evaluate has to be run first')

        if self.loo:
            self.evaluation_string = 'Accuracy: {accuracy:.2f}'.format(**self.results)
        else:
            self.evaluation_string = 'Accuracy: {balanced_accuracy:.2f}, AUC: {AUC:.2f}, F1-score: {F1:.2f}, Recall: ' \
                                     '{recall:.2f}, Precision: {precision:.2f}, Sensitivity: {sensitivity:.2f}, ' \
                                     'Specificity: {specificity:.2f}, ' \
                                     'PPV: {positive_predictive_value:.2f}, ' \
                                     'NPV: {negative_predictive_value:.2f}'.format(**self.results)
        print(self.evaluation_string)

    def set_evaluations(self):
        if self.loo:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('predictions_1st', return_prediction_first),
                                 ('predictions_2nd', return_prediction_second),
                                 ('true_1st', return_true_first),
                                 ('true_2nd', return_true_second)])
        else:
            evals = OrderedDict([('accuracy', accuracy_score),
                                 ('balanced_accuracy', balanced_accuracy),
                                 ('AUC', roc_auc_score),
                                 ('F1', f1_score),
                                 ('recall', recall_score),
                                 ('precision', precision_score),
                                 ('sensitivity', recall_score),           # recall is the same as sensitivity
                                 ('specificity', specificity),
                                 ('positive_predictive_value', ppv),
                                 ('negative_predictive_value', npv)])
        return evals
