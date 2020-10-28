# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by 6.034 Staff

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')


################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################


#### Part 1A: Classifying points ###############################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    return id_tree_classify_point(point, id_tree.apply_classifier(point))


#### Part 1B: Splitting data with a classifier #################################

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    dic={}
    for point in data:
        dic.setdefault(classifier.classify(point),[]).append(point)
    return dic    


#### Part 1C: Calculating disorder #############################################

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    dic = split_on_classifier(data, target_classifier)
    disorder=0
    T = len(data)
    for key in dic.keys():
        n = len(dic.get(key))
        disorder-=( (n/T) * log2(n/T) )
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    disorder=0
    T = len(data)
    dic=split_on_classifier(data, test_classifier)
    for key in dic.keys():
        disorder += branch_disorder(dic.get(key),target_classifier)*(len(dic.get(key))/T)
    return disorder    


## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:

# for classifier in tree_classifiers:
#     print(classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type")))


#### Part 1D: Constructing an ID tree ##########################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    best_disorder=INF
    best_classifier=None
    for test_classifier in possible_classifiers:
        cdis = average_test_disorder(data, test_classifier, target_classifier)
        dic=split_on_classifier(data, test_classifier)
        if (cdis<best_disorder)and(len(dic.keys())>1):
            best_disorder= cdis
            best_classifier=test_classifier
    if best_classifier==None:
        raise NoGoodClassifiersError
    else:
        return best_classifier


## To find the best classifier from 2014 Q2, Part A, uncomment:
# print(find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")))

def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node==None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    if branch_disorder(data, target_classifier)==0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
    else:
        try:
            bc = find_best_classifier(data, possible_classifiers, target_classifier)
        except NoGoodClassifiersError:
            return id_tree_node
        features = split_on_classifier(data, bc)
        id_tree_node.set_classifier_and_expand(bc, features.keys())
        branches = id_tree_node.get_branches()
        for k in branches.keys():
            branches[k] = construct_greedy_id_tree(features[k],[c for c in possible_classifiers if c!=bc],target_classifier,branches[k])
    return id_tree_node    
        
## To construct an ID tree for 2014 Q2, Part A:
# print(construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type")))

## To construct an ID tree for binary data for ANSWER_4 to 7
# print(construct_greedy_id_tree(binary_data,binary_classifiers,feature_test('Classification')))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
# tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
# print(id_tree_classify_point(tree_test_point, tree_tree))

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
# print(construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification")))
# print(construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class")))


#### Part 1E: Multiple choice ##################################################

ANSWER_1 = 'bark_texture'
ANSWER_2 = 'leaf_shape'
ANSWER_3 = 'orange_foliage'

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


#### OPTIONAL: Construct an ID tree with medical data ##########################

## Set this to True if you'd like to do this part of the lab
DO_OPTIONAL_SECTION = True

test_patient = {\
    'Age': 20, #int
    'Sex': 'M', #M or F
    'Chest pain type': 'atypical angina', #typical angina, atypical angina, non-anginal pain, or asymptomatic
    'Resting blood pressure': 100, #int
    'Cholesterol level': 120, #int
    'Is fasting blood sugar < 120 mg/dl': 'Yes', #Yes or No
    'Resting EKG type': 'ventricular hypertrophy', #normal, wave abnormality, or ventricular hypertrophy
    'Maximum heart rate': 150, #int
    'Does exercise cause chest pain?': 'Yes', #Yes or No
    'ST depression induced by exercise': 0, #int
    'Slope type': 'flat', #up, flat, or down
    '# of vessels colored': 0, #float or '?'
    'Thal type': 'fixed defect', #normal, fixed defect, reversible defect, or unknown
}

if DO_OPTIONAL_SECTION:
    from parse import *
    medical_id_tree = construct_greedy_id_tree(heart_training_data, heart_classifiers, heart_target_classifier_discrete)
    # print(medical_id_tree)
    # print(id_tree_classify_point(test_patient,medical_id_tree))

################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### Part 2A: Drawing Boundaries ###############################################

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### Part 2B: Distance metrics #################################################

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    n = min(len(u),len(v))
    dp = 0
    for i in range(n):
        dp += u[i]*v[i]
    return dp    

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    s = 0
    for i in range (len(v)):
        s += (v[i]**2)
    return (s**0.5)    

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    v = []
    for i in range (len(point1.coords)):
        v += [point1.coords[i]-point2.coords[i]]
    return norm(v)
    
def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    v = 0
    for i in range (len(point1.coords)):
        v += abs(point1.coords[i]-point2.coords[i])
    return v

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    v = 0
    for i in range (len(point1.coords)):
        if point1.coords[i] != point2.coords[i]:
            v += 1
    return v

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    v = []
    for i in range (len(point1.coords)):
        v += [point1.coords[i]]
    u = []
    for i in range (len(point2.coords)):
        u += [point2.coords[i]]
    if norm(v)!=0 and norm(u)!=0:
        return 1-(dot_product(u, v)/(norm(v)*norm(u)))    

    
#### Part 2C: Classifying points ###############################################

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    nn = sorted(data, key=lambda x: x.coords)
    nn = sorted(nn, key=lambda p: distance_metric(p,point))
    return nn[:k]

def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    points = get_k_closest_points(point, data, k, distance_metric)
    cs=[]
    for p in points:
        cs += [p.classification]
    return max(set(cs), key=lambda x:cs.count(x))

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
# print(knn_classify_point(knn_tree_test_point, knn_tree_data, 1, euclidean_distance))


#### Part 2C: Choosing k #######################################################

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    correct = 0
    num = len(data)
    for n in range (num):
        trainset = [data[i] for i in range (num) if i!=n]
        if knn_classify_point(data[n], trainset, k, distance_metric)==data[n].classification:
            correct+=1
    return correct/num        

def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    metrics = [euclidean_distance,manhattan_distance,hamming_distance,cosine_distance]
    bestm = metrics[0]
    bestk = 0
    bestcvf = 0
    for m in metrics:
        for k in range(1,len(data)+1):
            cvf = cross_validate(data, k, m)
            if cvf > bestcvf:
                bestcvf = cvf
                bestm = m
                bestk = k
    return (bestk, bestm)
                
## To find the best k and distance metric for 2014 Q2, part B, uncomment:
# print(find_best_k_and_metric(knn_tree_data))

#### Part 2E: More multiple choice #############################################

kNN_ANSWER_1 = 'Overfitting'
kNN_ANSWER_2 = 'Underfitting'
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3


#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
