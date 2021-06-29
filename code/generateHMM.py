import numpy as np
from argparse import ArgumentParser


## ------------------------ ##
## This program generates sequences of dice rolls according to the Hidden Markov Model 
## with parameters: 

##      - Fair state emission probabilities: equal (1/6)
##      - Loaded state emission probabilities: 1/10 for rolling 1-5, 1/2 for rolling 6
##      - Transition probabilities: 0.95 (F->F), 0.05 (F->L), 0.1 (L->F), 0.9 (L->L)

## Number of sequences, seed and printing of underlying states can be specified in commandline.
## ------------------------- ##


# get commandline parameters
parser = ArgumentParser(description="Generate HMM sequences")
parser.add_argument("-st",action="store", type=int, dest="states", default=2, help="Number of states (default = 2)", )
parser.add_argument("-se",action="store", type=int, dest="seed", help="Seed for reproducibility", )
parser.add_argument("-n",action="store", type=int, dest="n_seq", default = 100,  help="Number of sequences (default 100)", )
parser.add_argument("-ps",action="store_true", dest="print_state", default = True,  help="Print underlying states (default = False)", )
args = parser.parse_args()

states = args.states
n_seq = args.n_seq
seed = args.seed
symbols = "123456"
print_states = args.print_state

# Define probabilities for HMM
initial_prob = [1.0/states, 1.0/states]
transition_matrix = np.asarray([0.95, 0.05, 0.1, 0.9]).reshape(2,2)
fair_prob = [1.0/6, 1./6, 1./6, 1./6, 1./6, 1./6]
loaded_prob = [1./10, 1./10, 1./10, 1./10, 1./10, 5./10] 
emission_probs = np.array([fair_prob, loaded_prob])

np.random.seed(seed)

for _ in range(n_seq):
    length = 35
    sequence = ''
    state_seq = ''
    
    # Choose initial state
    coin = np.random.rand()
    prob = 0
    for state in range(states):
        prob += initial_prob[state]
        if coin <= prob:
            current_state = state
            break
    
    # Choose first number
    coin = np.random.rand()
    prob = 0
    for die in range(len(emission_probs[current_state])):
        prob += emission_probs[current_state, die]
        if coin <= prob:
            sequence += symbols[die]
            state_seq += str(current_state)
            break
            
    # Loop through model until full length sequence
    while len(sequence) < length:
        
        # Choose next state
        coin = np.random.rand()
        prob = 0
        for state in range(states):
            prob += transition_matrix[current_state, state]
            if coin <= prob:
                current_state = state
                break
        
        # Choose number in state
        coin = np.random.rand()
        prob = 0
        for die in range(len(emission_probs[current_state])):
            prob += emission_probs[current_state, die]
            if coin <= prob:
                sequence += symbols[die]
                state_seq += str(current_state)
                break
    
    if print_states:
        print(sequence)
        print(state_seq)
        print()
    else:
        print(sequence)
