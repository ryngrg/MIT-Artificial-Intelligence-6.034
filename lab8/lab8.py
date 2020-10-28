# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *


#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    ancestors=set()
    if net.get_parents(var)==set():
        return ancestors
    ancestors = ancestors.union(net.get_parents(var))
    for parent in net.get_parents(var):
        ancestors = ancestors.union(get_ancestors(net,parent))
    return ancestors

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    descendants=set()
    if net.get_children(var)==set():
        return descendants
    descendants = descendants.union(net.get_children(var))
    for child in net.get_children(var):
        descendants = descendants.union(get_descendants(net,child))
    return descendants

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    nondescendants = set()
    descendants = get_descendants(net, var)
    for v in net.get_variables():
        if (v!=var)and(v not in descendants):
            nondescendants.add(v)
    return nondescendants

#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    newgivens={}
    if (net.get_parents(var).issubset(givens.keys()))and(len(set(givens.keys()).intersection(get_descendants(net, var)))==0):
        for p in net.get_parents(var):
            newgivens[p]=givens[p]
        return newgivens
    else:
        return givens
    
def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"
    if givens!=None:
        givens = simplify_givens(net,list(hypothesis.keys())[0],givens)
    try:
        prob = net.get_probability(hypothesis,givens)
        return prob
    except ValueError:
        raise LookupError

def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"
    variables = list(reversed(net.topological_sort()))
    prob=1
    for i in range(len(variables)):
        hp={}
        gv={}
        hp[variables[i]] = hypothesis[variables[i]]
        for j in range(i+1,len(variables)):
            gv[variables[j]] = hypothesis[variables[j]]
        prob*=probability_lookup(net, hp, gv)
    return prob
    
def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"
    mprob=0
    for h in net.combinations(net.get_variables(),hypothesis):
        mprob+=probability_joint(net, h)
    return mprob

def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"
    if givens==None:
        return probability_marginal(net, hypothesis)
    for h in hypothesis.keys():
        if h in givens.keys():
            if hypothesis[h]!=givens[h]:
                return 0
    return probability_marginal(net, dict(hypothesis, **givens))/probability_marginal(net, givens)
    
def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net, hypothesis, givens)

#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    parameters=0
    for v in net.get_variables():
        if len(net.get_parents(v))==0:
            parameters+=len(net.get_domain(v))-1
        else:
            p = product([len(net.get_domain(p)) for p in net.get_parents(v)])
            parameters += (len(net.get_domain(v))-1)*p
    return parameters

#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    for c in net.combinations([var1, var2]):
        if givens is None:
            p1 = probability(net, {var1: c[var1]}, None)
            p2 = probability(net, {var1: c[var1]}, {var2: c[var2]})

        else:
            p1 = probability(net, {var1: c[var1]}, givens)
            p2 = probability(net, {var1: c[var1]}, dict(givens, **{var2: c[var2]}))
        if not(approx_equal(p1, p2)):
            return False
    return True
    
def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    #construct ancestral graph
    ancestors = get_ancestors(net, var1).union(get_ancestors(net,var2))
    if givens is None:
        g = []
    else:
        g = list(givens.keys())
        for v in g:
            ancestors = ancestors.union(get_ancestors(net, v))
    new_net = net.subnet(list(ancestors.union(set([var1, var2] + g))))
    
    #moralize by marrying parents
    for v in new_net.topological_sort():
        for p1 in new_net.get_parents(v):
            for p2 in new_net.get_parents(v):
                if p1!=p2:
                    new_net.link(p1,p2)
                
    #disorient by making edges undirected
    new_net = new_net.make_bidirectional()
    
    #remove givens
    for v in g:
        new_net.remove_variable(v)
        
    if new_net.find_path(var1, var2) is None:
        return True
    else:
        return False
    
#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
