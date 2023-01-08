from alphazero_nets import *
import random 
import pyspiel
import numpy as np



class Node:
    
    def __init__(self,state,obs,player_root_node,parent,prior_probability,player_turn,alphazero_network):
        '''
        Node class that describes a node in the MCTS tree. It contains:
            -state: the state of the node its a pyspiel state
            -obs : representation of the state of the board in tensor of shape (21,8,8)
            -player_root_node: the player that is playing the root node
            -parent: the parent node of the current node
            -prior_probability: the prior probability of the node
            -player_turn: the player that is playing the current node
            -alphazero_network: the neural network used to compute the prior probability and the value of the node
            -children: a dictionary that contains the children of the node
            -cput: the exploration parameter
            -is_leaf: boolean that indicates if the node is a leaf or not
            -visit_count: the number of times the node has been visited
            -tot_act_value: the total action value of the node
            -mean_value: the mean value of the node
        '''
        
        self.state = state
        self.obs = obs
        
        self.children = {}
        self.player_root_node = player_root_node
        self.parent = parent #[] if the node is the root node
        
        self.player_turn = player_turn
        self.neural_network = alphazero_network
        
        self.prior_probability = prior_probability
        self.visit_count = 0.
        self.tot_act_value = 0.
        self.mean_value = 0.
        
        
        self.is_leaf = True
        
        

    
    def expand(self):
        '''
            Expansion of the node by creating its children
        '''
        
        if (self.state.is_terminal()):
            raise Exception('Cannot expand a terminal node!!!')
        
        #Compute the prior probability of the children nodes
        prior_probability_child = self.neural_network.forward(self.obs)[0]
        prior_probability_child = prior_probability_child.flatten().detach().numpy() #convert to numpy array
        
        #Get the legal actions of the node
        legal_actions = self.state.legal_actions()
        
        #Loop over the legal actions and create the children nodes that come from the legal actions
        
        for id_action in range(len(legal_actions)):
            
            child_state = self.state.clone() #clone the state of the node
            
            child_state.apply_action(legal_actions[id_action]) #apply the action to the state
            
            if (not child_state.is_terminal()):
                obs_child = child_state.observation_tensor() #get the observation tensor of the child state
                
            else:
                obs_child = None  
                player_turn = None 
                   
            
            player_root_node = self.player_root_node #get the player that played the root node
                
            prior_probability_child_node = prior_probability_child[id_action] #get the prior probability of the child node
                
            player_turn = child_state.current_player() #get the player that is playing the child node
                
            self.children[id_action] = Node(child_state,obs_child,player_root_node,self,prior_probability_child_node,player_turn,self.neural_network) #create the child node
            
        self.is_leaf = False
        
        return
    
    
    
    def ucb_score(self,id_child,cput):
        
        '''
            Compute the ucb score of a child node given its id_child
        '''
        
        #Get the child node
        child_node = self.children[id_child]
        
        #Compute the ucb score
        ucb_score = child_node.mean_value + cput*child_node.prior_probability*np.sqrt(self.visit_count)/(1+child_node.visit_count)
        
        return ucb_score
     
        
    def select_child(self,cput):
        '''
            Select the child node with the highest ucb score
        '''
        #First check if there are children nodes
        
        if self.children == {}:
            raise Exception("The node has no children")
        
        else:
            #Get the ucb score of each child node
            nb_children = len(self.children)
            ucb_scores = [self.ucb_score(id_child,cput) for id_child in range(nb_children)]
            
            #Select the child node with the highest ucb score
            id_best_child = np.argmax(ucb_scores)
            
            return  self.children[id_best_child]
     
     
        
    def update_node(self,value):
        ''' 
            Update information of the node 
        '''
        #Update the node information
        
        self.visit_count += 1
        self.tot_act_value += value
        self.mean_value = self.tot_act_value/self.visit_count
        return
    
    
    
    
    
    def summary(self):
        '''
            Print the summary of the node
        '''
        print("Node summary...........")
        print("Player turn: ",self.player_turn)
        print("Visit count: ",self.visit_count)
        print("Mean action value: ",self.mean_value)
        print("Prior probability: ",self.prior_probability)
        print("Is leaf: ",self.is_leaf)
        print("Is terminal: ",self.state.is_terminal())
        print('------------------------------')
        return



    
    
    
def update_path(path,value):
        '''
            Update the information of each node from the current node to the root node
                Input:
                path = list of nodes from the current node to the root node
        '''
        #Convert value to a numpy array
        
        
        for node in path:
            node.update_node(value)
        return

















def test_node_functions():
    '''
        Test the functions of the node class
    '''
    neural_network = Alphazero_net()
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    obs_root = state.observation_tensor()
    root = Node(state,obs_root,neural_network)