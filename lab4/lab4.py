# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    varlist = csp.variables
    for var in varlist:
        if csp.get_domain(var) ==[]:
            return True
    return False    

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    constraints = csp.get_all_constraints()
    for c in constraints:
        if (csp.get_assignment(c.var1)!=None)and(csp.get_assignment(c.var2)!=None):
            if not(c.check(csp.get_assignment(c.var1),csp.get_assignment(c.var2))):
                return False
    return True            


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    extncount = 0
    while len(agenda)>0:
        extncount+=1
        nextprob = agenda.pop(0)
        if has_empty_domains(nextprob):
            continue
        if not(check_all_constraints(nextprob)):
            continue
        if nextprob.unassigned_vars==[]:
            # print(extncount)
            return (nextprob.assignments ,extncount)
        v = nextprob.pop_next_unassigned_var()
        cop=[]
        for val in nextprob.get_domain(v):
            cop.append(nextprob.copy().set_assignment(v,val))
        agenda = cop + agenda
    return (None, extncount)  


# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

# solve_constraint_dfs(get_pokemon_problem())

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################

def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    nbrs = csp.get_neighbors(var)
    nbrs_reduced = []
    for nbr in nbrs:
        rem=[]
        cons = csp.constraints_between(var, nbr)
        for w in csp.get_domain(nbr):
            setvval = csp.get_domain(var)
            for con in cons:
                flag = 0
                fineset=[]
                for vval in setvval:
                    if con.check(vval,w):
                        fineset+=[vval]
                        flag = 1
                setvval = fineset        
                if flag==0:
                    rem+=[w]
                    break
        if rem!=[]:
            nbrs_reduced += [nbr]
            for wval in rem:
                csp.eliminate(nbr,wval)
            if csp.get_domain(nbr)==[]:
                return None
    return sorted(nbrs_reduced)
   

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    extncount = 0
    while len(agenda)>0:
        extncount+=1
        nextprob = agenda.pop(0)
        if has_empty_domains(nextprob):
            continue
        if not(check_all_constraints(nextprob)):
            continue
        if nextprob.unassigned_vars==[]:
            # print(extncount)
            return (nextprob.assignments ,extncount)
        v = nextprob.pop_next_unassigned_var()
        cop=[]
        for val in nextprob.get_domain(v):
            copy_assigned = nextprob.copy().set_assignment(v,val)
            forward_check(copy_assigned, v)
            cop.append(copy_assigned)
        agenda = cop + agenda
    return (None, extncount) 


# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

# solve_constraint_forward_checking(get_pokemon_problem())

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue==None:
        queue = csp.get_all_variables()
    dequeued=[]
    while len(queue)>0:
        var = queue.pop(0)
        dequeued += [var]
        if csp.get_domain(var)==[]:
            return None
        next_check = forward_check(csp,var)
        if next_check == None:
            return None
        for n in next_check:
            if n not in queue:
                queue += [n]
    return dequeued            


# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

# solve_constraint_dfs(get_pokemon_problem())

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    extncount = 0
    while len(agenda)>0:
        extncount+=1
        nextprob = agenda.pop(0)
        if has_empty_domains(nextprob):
            continue
        if not(check_all_constraints(nextprob)):
            continue
        if nextprob.unassigned_vars==[]:
            # print(extncount)
            return (nextprob.assignments ,extncount)
        v = nextprob.pop_next_unassigned_var()
        cop=[]
        for val in nextprob.get_domain(v):
            copy_assigned = nextprob.copy().set_assignment(v,val)
            domain_reduction(copy_assigned, [v])
            cop.append(copy_assigned)
        agenda = cop + agenda
    return (None, extncount) 


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

# solve_constraint_propagate_reduced_domains(get_pokemon_problem())

ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue==None:
        queue = csp.get_all_variables()
    dequeued=[]
    while len(queue)>0:
        var = queue.pop(0)
        dequeued += [var]
        if csp.get_domain(var)==[]:
            return None
        next_check = forward_check(csp,var)
        if next_check == None:
            return None
        for n in next_check:
            if (enqueue_condition_fn(csp,n))and(n not in queue):
                queue += [n]
    return dequeued 

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    if len(csp.get_domain(var))==1:
        return True
    else:
        return False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    extncount = 0
    while len(agenda)>0:
        extncount+=1
        nextprob = agenda.pop(0)
        if has_empty_domains(nextprob):
            continue
        if not(check_all_constraints(nextprob)):
            continue
        if nextprob.unassigned_vars==[]:
            # print(extncount)
            return (nextprob.assignments ,extncount)
        v = nextprob.pop_next_unassigned_var()
        cop=[]
        for val in nextprob.get_domain(v):
            copy_assigned = nextprob.copy().set_assignment(v,val)
            if enqueue_condition!=None:
                propagate(enqueue_condition,copy_assigned, [v])
            cop.append(copy_assigned)
        agenda = cop + agenda
    return (None, extncount)


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

# solve_constraint_generic(get_pokemon_problem(), condition_singleton)

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if abs(m-n)==1:
        return True
    else:
        return False

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not(constraint_adjacent(m, n))

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    constlist=[]
    for i in range(len(variables)):
        for j in range(i+1,len(variables)):
            constlist += [Constraint(variables[i],variables[j],constraint_different)]
    return constlist        


#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 5
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
