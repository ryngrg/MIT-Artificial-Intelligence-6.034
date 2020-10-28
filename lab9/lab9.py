# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    n = len(training_points)
    point_to_weight={}
    for p in training_points:
        point_to_weight[p]=make_fraction(1/n)
    return point_to_weight

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    classifier_to_error_rate={}
    for c in classifier_to_misclassified.keys():
        e=0
        for p in classifier_to_misclassified[c]:
            e+=point_to_weight[p]
        classifier_to_error_rate[c]=e
    return classifier_to_error_rate

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    best=""
    if use_smallest_error:
        for c in classifier_to_error_rate.keys():
            if (best not in classifier_to_error_rate.keys()):
                best = c
            elif make_fraction(classifier_to_error_rate[c])<make_fraction(classifier_to_error_rate[best]):
                best = c
            elif (make_fraction(classifier_to_error_rate[c])==make_fraction(classifier_to_error_rate[best]))and(c<best):
                best = c
    else:
        for c in classifier_to_error_rate.keys():
            if (best not in classifier_to_error_rate.keys()):
                best = c
            elif make_fraction(abs(classifier_to_error_rate[c]-0.5))>make_fraction(abs(classifier_to_error_rate[best]-0.5)):
                best = c
            elif (make_fraction(abs(classifier_to_error_rate[c]-0.5))==make_fraction(abs(classifier_to_error_rate[best]-0.5)))and(c<best):
                best = c
    if make_fraction(classifier_to_error_rate[best])!=0.5:
        return best
    else:
        raise NoGoodClassifiersError
                
def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate==1:
        return -INF
    if error_rate!=0:
        return ln((1-error_rate)/error_rate)/2
    return INF

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclass=set()
    for p in training_points:
        cs=0
        for classifier in H:
            if p in classifier_to_misclassified[classifier[0]]:
                cs-=classifier[1]
            else:
                cs+=classifier[1]
        if cs<=0:
            misclass.add(p);
    return misclass

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    if len(get_overall_misclassifications(H, training_points, classifier_to_misclassified))>mistake_tolerance:
        return False
    return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    for p in point_to_weight.keys():
        if p not in misclassified_points:
            point_to_weight[p]=make_fraction(0.5*point_to_weight[p]/(1-error_rate))
        else:
            point_to_weight[p]=make_fraction(0.5*point_to_weight[p]/error_rate)
    return point_to_weight

#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    point_to_weight = initialize_weights(training_points)
    H=[]
    while max_rounds>0:
        classifier_to_error_rate = calculate_error_rates(point_to_weight, classifier_to_misclassified)
        try:
            best_classifier = pick_best_classifier(classifier_to_error_rate, use_smallest_error)
        except:
            return H
        error_rate = classifier_to_error_rate[best_classifier]
        voting_power = calculate_voting_power(error_rate)
        H.append((best_classifier, voting_power))
        point_to_weight = update_weights(point_to_weight, classifier_to_misclassified[best_classifier], error_rate)
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        if max_rounds!= INF:
            max_rounds -=1
    return H

#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
