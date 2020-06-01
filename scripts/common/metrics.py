import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
import seaborn as sns
sns.set()

# Accuracy
# def binary_acc(y_pred, y_test):
#     y_pred_tag = torch.round(y_pred)

#     correct_results_sum = (y_pred_tag == y_test).sum().float()
#     acc = correct_results_sum/y_test.shape[0]
#     acc = torch.round(acc * 100)
    
#     return acc

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag.view(-1) == y_test.view(-1)).sum().float()
    #print((y_pred_tag.view(-1) == y_test.view(-1)).size())
    #print(correct_results_sum)
    acc = correct_results_sum/y_test.view(-1).shape[0]
    acc = torch.round(acc * 100)
    return acc

# Accurate plotting of confusion matrix and full report of classification metrics
def plot_conf_matrix(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    cf = pd.DataFrame(confusion_matrix(y_true, y_pred))
    ax= plt.subplot()
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1.4)

    sns.heatmap(cf, annot=True, ax = ax,cmap='Blues',fmt='d')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_ylim([0,2])

# Intersection over Union for object detection
def inter_over_union_simple(y_pred, y_test, smooth = 1):
    intersection = torch.min(y_pred, y_test)
    union = y_pred + y_test - intersection
    iou = torch.mean(intersection/union)
    return iou * 100

# Intersection over Union for segmentation
def inter_over_union(y_pred, y_test, smooth = 1):
    y_pred_tag = torch.round(y_pred)
    
    conf_matrix = confusion_matrix(y_pred_tag.view(-1).cpu().detach().numpy(),y_test.view(-1).cpu().detach().numpy())
    #print(conf_matrix)
    intersection = np.diag(conf_matrix)
    predicted = conf_matrix.sum(axis = 1)
    target = conf_matrix.sum(axis = 0)
    union = predicted + target - intersection
    iou = np.mean(intersection/union)
    return iou * 100

# Dice coefficient
def dice_coef(y_pred, y_test, criterion, epsilon=1e-6):
    numerator = 2. * torch.sum(y_pred_thr * y_test, 1) + epsilon
    #denominator = torch.sum(torch.square(y_pred) + torch.square(y_test), 1) 
    denominator = torch.sum(y_pred + y_test,1)
    return 1 - torch.mean(numerator / (denominator + epsilon)) + criterion(y_pred, y_test)


