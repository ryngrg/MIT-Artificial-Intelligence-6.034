# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1
import time

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    chains = board.get_all_chains()
    for chain in chains:
        if len(chain)>3:
            return True
    if board.count_pieces()==42:
        return True
    return False        

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []
    boards=[]
    for i in range (board.num_cols):
        if not(board.is_column_full(i)):
            boards+=[board.add_piece(i)]
    return boards        

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if is_game_over_connectfour(board):
        for chain in board.get_all_chains():
            if len(chain)>3:
                if is_current_player_maximizer:
                    return -1000
                else:
                    return 1000
        return 0    

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    if is_game_over_connectfour(board):
        for chain in board.get_all_chains():
            if len(chain)>3:
                score = 1000 + ( (42-board.count_pieces())*20 )
                if is_current_player_maximizer:
                    return (-1*score)
                else:
                    return score
        return 0

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    if not(is_game_over_connectfour(board)):
        curlen=0
        opplen=0
        curnum=0
        oppnum=0
        for chain in board.get_all_chains(current_player=True):
            if len(chain)>1:
                curlen+=len(chain)
                curnum+=1
        for chain in board.get_all_chains(current_player=False):
            if len(chain)>1:
                opplen+=len(chain)
                oppnum+=1        
        if curnum==0:
            if oppnum==0:
                score=0
            else:
                score=-500
        else:
            if oppnum==0:
                score=500
            else:
                score = ((curlen/curnum)-(opplen/oppnum))*500
              
        if is_current_player_maximizer:
            return score
        else:
            return -1*score

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    agenda=[[state]]
    bestpath=[]
    statevals=0
    bestscore=-10000
    while len(agenda)>0:
        path = agenda.pop(0)
        if path[-1].is_game_over():
            if bestscore < path[-1].get_endgame_score():
                bestscore = path[-1].get_endgame_score()
                bestpath=path
            statevals+=1
            continue 
        newpaths = [path+[ext] for ext in path[-1].generate_next_states()]
        agenda = newpaths + agenda
    return (bestpath, bestscore, statevals)
    
# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

# pretty_print_dfs_type(dfs_maximizing(GAME1))

def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    if state.is_game_over():
        return [state], state.get_endgame_score(is_current_player_maximizer=maximize), 1
    else:
        nnodes = state.generate_next_states()
        path=[]
        bestscore=None
        stevals=0
        if maximize:
            for n in nnodes:
                pth, scr, evls= minimax_endgame_search(n,False)
                stevals+=evls
                if (bestscore==None)or(scr>bestscore):
                    bestscore=scr
                    path=[state]+pth
        else:            
            for n in nnodes:
                pth, scr, evls = minimax_endgame_search(n,True)
                stevals+=evls
                if (bestscore==None)or(scr<bestscore):
                    bestscore=scr
                    path=[state]+pth
        return path, bestscore, stevals            

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    if state.is_game_over():
        return [state], state.get_endgame_score(is_current_player_maximizer=maximize), 1
    if depth_limit==0:
        return [state], heuristic_fn(state.get_snapshot(),maximize), 1
    if depth_limit>0:
        nnodes = state.generate_next_states()
        path=[]
        bestscore=None
        stevals=0
        if maximize:
            for n in nnodes:
                pth, scr, evls= minimax_search(n,heuristic_fn,depth_limit-1,False)
                stevals+=evls
                if (bestscore==None)or(scr>bestscore):
                    bestscore=scr
                    path=[state]+pth
        else:            
            for n in nnodes:
                pth, scr, evls = minimax_search(n,heuristic_fn,depth_limit-1,True)
                stevals+=evls
                if (bestscore==None)or(scr<bestscore):
                    bestscore=scr
                    path=[state]+pth
        return path, bestscore, stevals
        
# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    if state.is_game_over():
        return [state], state.get_endgame_score(maximize), 1
    if depth_limit==0:
        return [state], heuristic_fn(state.get_snapshot(),maximize), 1
    if depth_limit>0:
        nnodes = state.generate_next_states()
        path=[]
        bestscore=None
        stevals=0
        if maximize:
            for n in nnodes:
                pth, scr, evls= minimax_search_alphabeta(n,alpha,beta,heuristic_fn,depth_limit-1,False)
                stevals+=evls
                if (bestscore==None)or(scr>bestscore):
                    bestscore=scr
                    path=[state]+pth
                alpha=max(alpha,bestscore)
                if alpha>=beta: 
                    break
        else:            
            for n in nnodes:
                pth, scr, evls = minimax_search_alphabeta(n,alpha,beta,heuristic_fn,depth_limit-1,True)
                stevals+=evls
                if (bestscore==None)or(scr<bestscore):
                    bestscore=scr
                    path=[state]+pth
                beta=min(beta,bestscore)
                if beta<=alpha:
                    break
        return path, bestscore, stevals

# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value=AnytimeValue()
    for i in range (1,depth_limit+1):
        res = minimax_search_alphabeta(state,-INF,INF,heuristic_fn,i,maximize)
        anytime_value.set_value(res)
    return anytime_value

########## prog deepening with time limit ##########
#def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
#                          maximize=True, time_limit=INF) :
#    start = time.time()
#    anytime_value=AnytimeValue()
#    for i in range (1,depth_limit+1):
#        res = minimax_search_alphabeta(state,-INF,INF,heuristic_fn,i,maximize)
#        anytime_value.set_value(res)
#        if (time.time()-start)>=time_limit:
#            break
#    return anytime_value    


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=10, time_limit= ).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = "heuristic_connectfour"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
