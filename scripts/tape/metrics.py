from typing import Sequence, Union
import numpy as np
import scipy.stats
from sklearn.metrics import classification_report, confusion_matrix

from .registry import registry


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total

@registry.register_metric('recall')
def recall(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        print('sequence case')
        true_pos_number = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            true_pos = label_array[mask] == 1
            pos = sum(true_pos)
            true_pos_num = sum(pred_array[mask][true_pos] == 1)
            
            true_pos_number += true_pos_num
            total += pos
        return true_pos_number / total
        
@registry.register_metric('false_neg')
def false_neg(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        #print('sequence case')
        false_neg_number = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            true_pos = label_array[mask] == 1
            pos = sum(true_pos)
            false_neg = sum(pred_array[mask][true_pos] == 0)
            
            false_neg_number += false_neg 
        return false_neg_number
        
        
@registry.register_metric('iou')
def inter_over_union(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
                 
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        
        label_array = np.asarray(target)
        pred_array = np.asarray(prediction).argmax(-1)
        mask = label_array != -1
        conf_matrix = confusion_matrix(label_array[mask],pred_array[mask])
        #print(conf_matrix)
        #plot_conf_matrix(y_pred_tag.view(-1).detach().numpy(),y_test.view(-1).detach().numpy())
        intersection = np.diag(conf_matrix)
        predicted = conf_matrix.sum(axis = 1)
        target = conf_matrix.sum(axis = 0)
        union = predicted + target - intersection
        iou = np.mean(intersection/union)
        return iou * 100