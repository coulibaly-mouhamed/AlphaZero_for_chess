'''
Implement a Monte Carlo Tree Search algorithm for the game of chess.
'''
import torch

class Node:
    
    def __init__(self,state,prior_probability,player_turn):
        '''
        Node class that describes a node in the MCTS tree. It contains:
            -state: the state of the board
                *type : torch.tensor of size (22,8,8)
            -Prior_probability: prior probbabilty for this stata predicted by the neural network for example for a given root node the children nodes with highest prior probability will be more likely to be explored
                *Prior probability has the same type as the policy head output of the neural network (torch.tensor of size 4672=73*8*8)
            -player_turn: the player turn
                *type : int
            -children: a dictionnary of children nodes
                *type : dict
            -value: the value of the node
                *type :float between -1 and 1
            -visit_count: the number of times the node has been visited
                *type : int       
        '''
        self.state = state
        self.prior_probability = prior_probability
        self.player_turn = player_turn
        self.children = {}
        self.value = 0
        self.visit_count = 0
    
    def expand(self,prior_probability):
        '''
        Expand the node by adding a child node to the node
         prior_probability: tensor of size (4672= 73*8*8) containing the prior probability for each possible move
        '''
        n_actions = 4672 # 8*8*73 number of possible actions
        #Loop over prior probability to find the moves that have a non zero probability
        for id_action in range (n_actions):
            if prior_probability[id_action] != 0:
                #State of the board coming from the action
                state_child = 
                
        
        pass

class Tree:
    '''
    Tree of  possibilies in chess game 
    '''
    
    def __init__(self, root):
        self.root = root
        self.children = []
        self.visits = 0
        self.value = 0
        
    def expand(self, children):
        self.children = children
        
    def evaluate(self,neural_net):
        self.value = neural_net(self.root)
        self.visits += 1