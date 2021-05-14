# MASIW - Meta Subsampling Importance Weighting
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import math
import time
import argparse
from collections import Counter, deque, OrderedDict

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network
from maml import MAML
from utils import *

#set reproducibility
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

def label_shift(args:dict):
    """ 
    Runs main label shift experiment for specific source and target distribution 
    @Returns:
    - RESULTS : 
    {
        'naive' : None, #default training
        'bbse'  : None, #IW only
        'malls' : None,  #subsampling + importance weighting
        'masiw' : None, #meta-learning + subsampling + importance weighting
    }
    """
    RESULTS = {}
        
    
    X, y = load_digits(return_X_y=True) #multiclassification
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.target_ratio, random_state=42)
    
    
    #### --- Create Imbalanced Dataset
    
    idx_by_label = group_by_label(y_train) #label : [indices of all labels]
    
    #Source distribution shift
    size = 2 * X_train.shape[0]
    shifted_dist_idx = dirichlet_distribution(
        alpha=args.source_alpha, idx_by_label=idx_by_label, size=size, no_change=args.keep_source)
    
    #Test distribution shift
    idx_by_label = group_by_label(y_test) #label : [indices of all labels]
    size = 2 * X_test.shape[0]
    shifted_test_dist_idx = dirichlet_distribution(
        alpha=args.target_alpha, idx_by_label=idx_by_label, size=size, no_change=args.keep_target)
    
    #train Distribution shift
    plot(y_train, shifted_dist_idx, 'Train', args.display_plots)
    
    #test Distribution shift
    plot(y_test, shifted_test_dist_idx, 'Test', args.display_plots)
    
    #### --- Sync With Data

    ### No subsampling - take source Dist.
    X_train, y_train = X_train[shifted_dist_idx], y_train[shifted_dist_idx]

    ### Shifting test distribution
    X_test, y_test = X_test[shifted_test_dist_idx], y_test[shifted_test_dist_idx]
    
    #Get source (train) and target (test) label distributions
    dist_train = get_distribution(y_train)
    dist_test  = get_distribution(y_test)

    #### --- Train Naive Model
    
    ##typecast to tensors
    X_train = torch.DoubleTensor(X_train).to(device)
    X_test = torch.DoubleTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)
    
    
    model_normal, cost, training_accuracy, test_accuracy = train(
        (X_train, y_train, X_test, y_test), print_st=args.display_plots, epochs=args.epochs)
    
    #graph cost
    plot_cost(cost, training_accuracy, test_accuracy, 'Full Batch Training Cost', args.display_plots)
        
    #### --- Test Model
    
    ### Estimated distribution
    score, predictions = predict(model_normal, (X_test, y_test))
    
    RESULTS['naive'] = score
    
    if args.alg == 'naive':
        return RESULTS
    
    #### -- MALLS Subsampling
    
    # Generate Medial Distribution
    biased_probs = 1. / np.array(list(dist_train.values()))
    biased_probs /= np.sum(biased_probs)
    
    p = np.zeros(y_train.shape)

    for i in range(len(p)):
        p[i] = biased_probs[y_train[i]]

    p /= p.sum() #normalize
    medial_idx = np.random.choice(np.arange(len(y_train)), size=y_train.shape, replace=True, p=p)
    
    
    if args.display_plots:
        ### Medial Distribution
        plt.bar(
            x=np.unique(y_train[medial_idx]), height=get_distribution(y_train[medial_idx].numpy()).values())
        plt.title("Medial Distribution")
        plt.xlabel("Class label")
        plt.ylabel("PMF")
        plt.grid()
        plt.show()
    
    if args.alg != 'bbse':
        ### Subsampling - take Medial Dist.
        X_train, y_train = X_train[medial_idx], y_train[medial_idx]
    
    ### --- BBSE Label Shift
    data = X_train.clone(), y_train.clone() #store original training distribution.

    #Split training into training (source) and validation (hold-out)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=args.validation_ratio, random_state=42)
    
    ### obtain classifier by training on X_train, y_train
    f, cost, training_accuracy, test_accuracy = train(
        (X_train, y_train, X_test, y_test), print_st=args.display_plots, epochs=args.epochs)
    
    
    #graph cost
    plot_cost(cost, training_accuracy, test_accuracy, 'Source only Cost', args.display_plots)
    
    ### --- Generate Label Shift
    conf_matrix, k = calculate_confusion_matrix(X_validation, y_validation, f)
    mu = calculate_target_priors(X_test, k, f)
    #generate label weights, if possible
    label_weights = compute_weights(conf_matrix, mu, args.delta)
    
    #### --- Importance Weighting Training
    X_train, y_train = data #regain data
    f_weighted, cost, training_accuracy, test_accuracy = train_iw(
        (X_train, y_train, X_test, y_test), label_weights, f, epochs=args.epochs, print_st=args.display_plots)
    
    plot_cost(cost, training_accuracy, test_accuracy, 'Full Source Training Cost', args.display_plots)
    
    ### --- Importance Weighting Test
    score, _ = predict_IW(f_weighted, label_weights, (X_test, y_test)) ### Prediction

    if args.alg != 'masiw':
        RESULTS[args.alg] = score
        return RESULTS

    #### --- MAML - Importance Weight Bias Reduction
    maml = MAML(X_validation, y_validation, f_weighted, label_weights) # declare maml

    for _ in range(args.meta_updates):
        maml.update()
    
    label_weights = maml.get_label_weights()   
    score, _ = predict_IW(f_weighted, label_weights, (X_test, y_test))
    
    RESULTS['masiw'] = score
    
    return RESULTS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Efficient Label Shift based Domain Adaption')

    #source alpha
    parser.add_argument('-source_alpha', metavar='Dirichlet Distibution parameter', type=float, default=1,
                        help='Magnitude of Label Shift based on dirichlet dist.')
    #keep source alpha
    parser.add_argument('-keep_source', type=bool, default=True,
                        help='Keep original distribution or dirichlet label shift simulation')
    #target distribution alpha
    parser.add_argument('-target_alpha', metavar='Dirichlet Distibution parameter', type=float, default=1,
                        help='Magnitude of Label Shift based on dirichlet dist.')
    #keep target alpha
    parser.add_argument('-keep_target', type=bool, default=True,
                        help='Keep original distribution or dirichlet label shift simulation')
    #target distribution ratio
    parser.add_argument('-target_ratio', type=float, default=0.2,
                        help='Proportion of original data to be set aside for target set')
    #validation distribution ratio
    parser.add_argument('-validation_ratio', type=float, default=0.5,
                        help='Proportion of train data to be set aside for holdout set')
    #BBSE delta
    parser.add_argument('-delta', type=float, default=1e-3,
                        help='BBSE inverse confusion matrix threshold paramter, delta.')
    #display graphs
    parser.add_argument('-display_plots', type=bool, default=False,
                        help='Show graphs when running experiment?')
    #algorithm type
    parser.add_argument('-alg', metavar='Label Shift Algorithm', type=str, default='MASIW',
                        help='Type of Label Shift algorithm to run')
    #number of epochs to run algs
    parser.add_argument('-epochs', metavar='Gen epoch count', type=int, default=350,
                        help='Number of epochs to run all learning algorithms')
    #number of MAML updates
    parser.add_argument('-meta_updates', metavar='number of meta updates', type=int, default=2,
                        help='Number of MAML meta updates. Note: More updates increases sensitivity')
    
    
    args = parser.parse_args() #parse arguments
    
    #Run MASIW
    result = label_shift(args)
    
    print(f"{args.alg} result : {result[args.alg]}")