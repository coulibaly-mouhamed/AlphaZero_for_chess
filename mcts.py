'''
Implement a Monte Carlo Tree Search algorithm for the game of chess.
'''
from statistics import mode

from alphazero_nets import *
import torch
from chess_board_gym import *
import copy 
import numpy as np 
import multiprocessing as mp




class Node:
    
    def __init__(self,env,state,alphazero_network,player_turn=1,prior_probability=1,action_past=None):
        '''
        Node class that describes a node in the MCTS tree. It contains:
            -env: the gym environment of the game
            -state: the state of the board
                *type : torch.tensor of size (22,8,8)
            -Prior_probability: Probability to take the action which leads to this node from the parent node
                *type: float 
            -player_turn: the player turn
                *type : int
            -children: a dictionnary of children nodes
                *type : dict
            -visit_count: the number of times the node has been visited
                *type : int 
            -tot_action_value: the total value of the action
                *type : float
            -mean_action_value: the mean value of the action
                *type : float   
            -actions_list: the list of legal actions for the node
            -parent: the parent node
            -cput: the exploration parameter
            -alphazero_network: the neural network that will be used to predict the prior probability of the children nodes and the value of the state to guide the search     
        '''
        self.env = env
        self.state = state
        self.prior_probability = prior_probability
        self.player_turn = player_turn
        self.children = {}
        self.visit_count = 0
        self.tot_action_value = 0
        self.mean_action_value = 0
        #self.actions_list = self.env.legal_actions
        self.neural_network = alphazero_network
        self.cput = 1
        self.is_leaf = True
        self.action_past = action_past #Action that has been taken to reach the node
        self.action_tracker = []
    
    def create_children(self,id_action,prior_probability_child):
        
        child_env = gym.make('Chess-v0')
        child_env = BoardEncoding(child_env,history_length=1)
        child_env = MoveEncoding(child_env)
        child_env.reset()
        
        if len(self.action_tracker) != 0 :
            for act_past in self.action_tracker:
                #Update the new environment with the history of actions
                child_env.step(act_past)
       
        state_child, reward, done, info = child_env.step(self.env.legal_actions[id_action])
        
        #Convert prior_probability_child to a numpy array
        self.children[id_action] = Node(child_env,state_child,self.neural_network,-self.player_turn,prior_probability_child[self.env.legal_actions[id_action]],self.env.legal_actions[id_action])
        
        #Add the parent node to the child node
        self.children[id_action].action_tracker = copy.deepcopy(self.action_tracker)
        self.children[id_action].action_tracker.append(self.env.legal_actions[id_action])
        
        return 
    
    def expand(self):
        '''
        Expand the node by adding a child node to the node
         prior_probability: tensor of size (4672= 73*8*8) containing the prior probability for each possible move
        '''
        #Compute the prior probability of the children nodes
        prior_probability_child = self.neural_network.forward(self.state)[0]
        prior_probability_child = prior_probability_child.flatten().detach().numpy()
       
        '''
        id_actions =[i for i in range(len(self.env.legal_actions))]
        try:
            with mp.Pool(2) as pool:
                childrens = pool.map(self.create_children,id_actions)
            self.children = {i:childrens[i] for i in range(len(self.env.legal_actions))}
        
        except:
        '''
        
        for id_action in range(len(self.env.legal_actions)):
            self.create_children(id_action,prior_probability_child)
            
        self.is_leaf = False
        
        return
    
    

    def select_child(self):
        '''
          Select the node with the highest UCB score
            -Outputs: Node of the best child
        '''
        
        best_score = float('-inf')
        best_action = None
        for id_action in range(len(self.env.legal_actions)):
            score = self.ucb_score(id_action)
            if score > best_score:
                best_score = score
                best_action = id_action
        return self.children[best_action]
        
        

    def ucb_score(self,id_action):
        '''
         Compute the UCB score for the node
        '''
        U_s_a = self.cput*self.prior_probability*np.sqrt(self.visit_count)/(1+ self.children[id_action].visit_count)
        #U_s_a = self.cput*np.sqrt(self.sum_visits)/(1+self.children[id_action].visit_count)    
        
        return self.mean_action_value + U_s_a
        
    
    def update_value(self,value):
        '''
            Update the state of the tree after a simulation
        '''
       
        
        self.visit_count += 1
        self.tot_action_value += value*self.player_turn
        
       
        
        self.mean_action_value = self.tot_action_value/self.visit_count
        return


def compute_pi_posterior(root_node,temperature):
    '''
    Compute the posterior probability of the root node
        root_node: the root node of the tree
        temperature: the temperature parameter
    
    '''
    inv_temp = 1/temperature
    pi = torch.zeros(4672)
    actions_list = root_node.env.legal_actions
    for id_action in range(len(actions_list)):
        pi[actions_list[id_action]] = root_node.children[id_action].visit_count#**inv_temp
        pi[actions_list[id_action]]= pi[actions_list[id_action]]/(root_node.visit_count)#**inv_temp)
        root_node.children[id_action].env.close()
    #Reset and delete the children nodes
    
    del root_node.children 
    return pi
  
    
def update_value(search_path,value):
    '''
    Update the value of the nodes in the search path
        search_path: the list of nodes in the search path
        value: the value of the state
    '''
    for node in search_path:
        node.update_value(value)
    return


def MCTS(root,num_simulations,actions_tracker):
    '''
    Perform a Monte Carlo Tree Search algorithm for the game of chess
    root: the root node of the tree
    num_simulations: the number of simulations to perform
    actions_tracker: the list of actions that have been taken to reach the root node
    '''
    root.action_tracker = actions_tracker
    
    #First Expansion
    root.expand()
    
    for i in range(num_simulations):
       
        node = root
        search_path = [node]
        
        #Selection
        while(not node.is_leaf):
            
            node = node.select_child()
            search_path.append(node)
            
        #Expansion
     
        if (node.visit_count==0):
           
            prior_prob,value = node.neural_network.forward(node.state) #rollout
            
            update_value(search_path,value)#Backpropagation
        else:
            
            node.expand()
            
            #Simulation
            node = node.select_child()
            
            search_path.append(node)
    
            prior_prob,value = node.neural_network.forward(node.state) #rollout
           
            update_value(search_path,value) #Backpropagation
       
        
    temperature = 1
    
    pi =compute_pi_posterior(root,temperature)   
    return pi 



if __name__ == '__main__':
    
    neural_network = Alphazero_net()
    env = gym.make('Chess-v0')
    env = BoardEncoding(env,history_length=1)
    env = MoveEncoding(env)
    env.reset()
    env_2 = game(env,1)
    print('Playing....')
    env_3,board = env_2.play(5)
    print('Done')
    #print(env.render(mode='unicode'))
    #print('legal actions ',len(env.legal_actions))
    
    root = Node(env_3,board,neural_network,1)
    pi = MCTS(root,1000)
    #print("Pi: ",pi)
    #print('Pi shape: ',pi.shape)
    #print('Visit count: ',root_final.visit_count)
    #print('Sum visits: ',root_final.sum_visits)
    #print('Mean action value: ',root_final.mean_action_value)
    #print('Children: ',len(root_final.children))
    
    '''
    pi_legal_actions = []
    for actions in env.legal_actions:
       pi_legal_actions.append(pi[actions])
    #Convert to numpy array
    print('Legal actions:',env.legal_actions)
    print('Pi legal actions: ',pi_legal_actions)
    '''
    #Print the mean_action_value of the children nodes
    '''
    for id_action in range(len(root_final.actions_list)):
        print('Action: ',root_final.actions_list[id_action])
        print('Pi: ',id_action,':',pi[root_final.actions_list[id_action]])
        print('Mean action value for action ',id_action,' : ',root_final.children[id_action].mean_action_value)
        print('Visit count for action ',id_action,' : ',root_final.children[id_action].visit_count)
        #print('Prior probability for action ',id_action,' : ',root_final.children[id_action].prior_probability)
        try:
            print('UCB score for action ',id_action,' : ',root_final.children[id_action].ucb_score(id_action))
        except:
            print('Problem with UCB score')
            print('Visit count',root_final.children[id_action].visit_count)
            print('Prior probability',root_final.children[id_action].prior_probability)
        print('-------------------------')
    
    print('Visit count: ',root_final.visit_count)
    '''