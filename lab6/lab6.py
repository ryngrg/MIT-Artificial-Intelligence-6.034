# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x>=threshold:
        return 1
    else:
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+(e**(steepness*(midpoint-x))))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return (-0.5)*((desired_output-actual_output)**2)


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    neurons = net.topological_sort()
    dic = {}
    for n in neurons:
        ins = net.get_incoming_neighbors(n)
        s = 0
        for i in ins:
            w = net.get_wires(i, n)[0].get_weight()
            s += node_value(i,input_values,dic) * w
        dic[n] = threshold_fn(s)
    return (dic[neurons[-1]], dic)    

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    steps = [step_size,(-1*step_size),0]
    bestval = -INF
    ins = inputs.copy()
    for s0 in steps:
        for s1 in steps:
            for s2 in steps:
                val = func((inputs[0]+s0),(inputs[1]+s1),(inputs[2]+s2))
                if val > bestval:
                    bestval = val
                    ins[0]=inputs[0]+s0
                    ins[1]=inputs[1]+s1
                    ins[2]=inputs[2]+s2
    return (bestval, ins)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    s ={wire}
    s.add(wire.startNode)
    s.add(wire.endNode)
    for w in net.get_wires(wire.endNode):
        s = s.union(get_back_prop_dependencies(net, w))
    return s

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    dic={}
    nodes=net.topological_sort()
    for i in reversed( range(len(nodes)) ):
        n = nodes[i]
        if net.is_output_neuron(n):
            dic[n] = neuron_outputs[n]*(1-neuron_outputs[n])*(desired_output-neuron_outputs[n])
        else:
            s=0
            for a in net.get_outgoing_neighbors(n):
                s += net.get_wires(n,a)[0].get_weight() * dic[a]
            dic[n] = neuron_outputs[n]*(1-neuron_outputs[n])*s
    return dic        
            
def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    deltas = calculate_deltas(net, desired_output, neuron_outputs)
    wires = net.get_wires()
    for w in wires:
        ow = w.get_weight()
        w.set_weight(ow + (r*node_value(w.startNode,input_values,neuron_outputs)*deltas[w.endNode]))
    return net    

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    actual_output, neuron_outputs = forward_prop(net, input_values, sigmoid)
    iterations = 0
    while minimum_accuracy > accuracy(desired_output, actual_output):
        iterations += 1
        net = update_weights(net, input_values, desired_output, neuron_outputs, r)
        actual_output, neuron_outputs = forward_prop(net, input_values, sigmoid)
    return (net, iterations)    

#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 36
ANSWER_2 = 11
ANSWER_3 = 8
ANSWER_4 = 115
ANSWER_5 = 30

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small','medium','large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A','C']
ANSWER_12 = ['A','E']


#### SURVEY ####################################################################

NAME = "Aaryan Garg"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = "everything"
WHAT_I_FOUND_BORING = "nothing"
SUGGESTIONS = None
