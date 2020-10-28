# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    ans=0
    for i in range(len(u)):
        ans+=u[i]*v[i]
    return ans

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    s=0
    for i in range(len(v)):
        s+=v[i]**2
    return s**0.5

#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w,point.coords)+svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    if positiveness(svm, point)==0:
        return 0
    if positiveness(svm, point)>0:
        return 1
    return -1

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    bad=set()
    for point in svm.support_vectors:
        if positiveness(svm,point)!=point.classification:
            bad.add(point)
    for point in svm.training_points:
        if abs(positiveness(svm,point))<1:
            bad.add(point)
    return bad

#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    bad=set()
    for point in svm.training_points:
        if point.alpha<0:
            bad.add(point)
        if point not in svm.support_vectors:
            if point.alpha!=0:
                bad.add(point)
    for point in svm.support_vectors:
        if point.alpha<=0:
            bad.add(point)
    return bad

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    cond4 = False
    cond5 = False
    s=0
    for point in svm.training_points:
        s+=point.classification*point.alpha
    if s==0:
        cond4 = True
    v=[]
    for i in range(len(svm.w)):
        s=0
        for point in svm.training_points:
            s+=point.classification*point.alpha*point.coords[i]
        v.append(s)
    if v==svm.w:
        cond5 = True
    return cond4 and cond5

#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    bad = set()
    for point in svm.training_points:
        if classify(svm,point)!=point.classification:
            bad.add(point)
    return bad

#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    v=[]
    for point in svm.training_points:
        if point.alpha>0:
            v.append(point)
    svm.support_vectors=v
    v=[]
    for i in range(2):
        s=0
        for point in svm.training_points:
            s+=point.classification*point.alpha*point.coords[i]
        v.append(s)
    svm.w = v
    posb=[]
    negb=[]
    for point in svm.support_vectors:
        if point.classification==1:
            posb.append(1-dot_product(svm.w,point.coords))
        if point.classification==-1:
            negb.append(-1-dot_product(svm.w,point.coords))
    svm.b=(max(posb)+min(negb))/2
    return svm

#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A','D']
ANSWER_6 = ['A','B','D']
ANSWER_7 = ['A','B','D']
ANSWER_8 = []
ANSWER_9 = ['A','B','D']
ANSWER_10 = ['A','B','D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
