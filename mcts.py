'''
Implement a Monte Carlo Tree Search algorithm for the game of chess.
'''
from statistics import mode
from alphazero_nets import *
import torch
from chess_board_gym import *
import copy 
import numpy as np 

def get_prior_probability(alphazero_network,state):
    '''
    Get the prior probability of the actions from the neural network
    alphazero_network: the neural network
    state: the state of the board
    '''
    policy, value = alphazero_network.forward(state)
    return policy


class Node:
    
    def __init__(self,env,state,alphazero_network,player_turn,prior_probability=1):
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
        self.actions_list = env.legal_actions
        self.neural_network = alphazero_network
        self.sum_visits = 0
        self.cput = 1
        self.parent = None
        self.is_leaf = True
    
    
    def expand(self):
        '''
        Expand the node by adding a child node to the node
         prior_probability: tensor of size (4672= 73*8*8) containing the prior probability for each possible move
        '''
        #Compute the prior probability of the children nodes
        prior_probability_child = self.neural_network.forward(self.state)[0]
        prior_probability_child = prior_probability_child.flatten().detach().numpy()
        for id_action in range(len(self.actions_list)):
            child_env = copy.deepcopy(self.env)
            state_child, reward, done, info = child_env.step(self.actions_list[id_action])
            #Convert prior_probability_child to a numpy array
            self.children[id_action] = Node(child_env,state_child,self.neural_network,self.player_turn,prior_probability_child[id_action])
            #Add the parent node to the child node
            self.children[id_action].parent = self
        self.is_leaf = False
        return

    def select_child(self):
        '''
          Select the node with the highest UCB score
            -Outputs: Node of the best child
        '''
        self.compute_sum_visits()
        best_score = float('-inf')
        best_action = None
        for id_action in range(len(self.actions_list)):
            score = self.ucb_score(id_action)
            if score > best_score:
                best_score = score
                best_action = id_action
        return self.children[best_action]
        
        
    def compute_sum_visits(self):
        '''
            Compute the sum of the visits of the children nodes
        '''
        for id_action in range(len(self.actions_list)):
            self.sum_visits += self.children[id_action].visit_count
            
        return
    
    def ucb_score(self,id_action):
        '''
         Compute the UCB score for the node
        '''
        #U_s_a = self.cput*self.prior_probability*np.sqrt(self.sum_visits)/(1+self.children[id_action].visit_count) 
        U_s_a = self.cput*np.sqrt(self.sum_visits)/(1+self.children[id_action].visit_count)    
        #print('Sum visits',self.sum_visits)
        return self.mean_action_value + U_s_a
        
    
    def update_value(self,value):
        '''
            Update the state of the tree after a simulation
        '''
        self.visit_count += 1
        self.tot_action_value += value
        self.mean_action_value = self.tot_action_value/self.visit_count
        parent  = self.parent
        while(parent != None):
            parent.visit_count += 1
            parent.tot_action_value += value
            parent.mean_action_value = self.parent.tot_action_value/self.parent.visit_count
            parent = parent.parent
        return


def compute_pi_posterior(root_node,temperature):
    '''
    Compute the posterior probability of the root node
        root_node: the root node of the tree
        temperature: the temperature parameter
    
    '''
    inv_temp = 1/temperature
    pi = torch.zeros(4672)
    root_node.compute_sum_visits() #Compute the sum of the visits of the children nodes
    for id_action in range(len(root_node.actions_list)):
        pi[id_action] = root_node.children[id_action].visit_count**inv_temp
        pi[id_action]= pi[id_action]/(root_node.sum_visits**inv_temp)
    return pi
    


def MCTS(root,num_simulations):
    '''
    Perform a Monte Carlo Tree Search algorithm for the game of chess
    root: the root node of the tree
    num_simulations: the number of simulations to perform
    '''
    root.expand()
    for i in range(num_simulations):
        node = root
        #Selection
        while(not node.is_leaf):
            node = node.select_child()
        #Expansion
        if (node.visit_count==0):
            prior_prob,value = node.neural_network.forward(node.state) #rollout
            node.update_value(value) #Backpropagation
        else:
            node.expand()
            #Simulation
            node = node.select_child()
            prior_prob,value = node.neural_network.forward(node.state) #rollout
            node.update_value(value) #Backpropagation
        
    temperature = 1
    pi =compute_pi_posterior(root,temperature)   
    return root,pi 



if __name__ == '__main__':
    neural_network = Alphazero_net()
    env = gym.make('Chess-v0')
    env = BoardEncoding(env,history_length=1)
    env = MoveEncoding(env)
    env.reset()
    env_2 = game(env,1)
    print('Playing....')
    env_2.play(10)
    print('Done')
    print(env.render(mode='unicode'))
    #print('legal actions ',len(env.legal_actions))
    
    root = Node(env,env.reset(),neural_network,1)
   
    root_final,pi = MCTS(root,500)
    print("Pi: ",pi)
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
    
    for id_action in range(len(root_final.actions_list)):
        print('Action: ',root_final.actions_list[id_action])
        print('Pi: ',id_action,':',pi[id_action])
        print('Mean action value for action ',id_action,' : ',root_final.children[id_action].mean_action_value)
        print('Visit count for action ',id_action,' : ',root_final.children[id_action].visit_count)
        print('Sum visits for action ',id_action,' : ',root_final.children[id_action].sum_visits)
        #print('Prior probability for action ',id_action,' : ',root_final.children[id_action].prior_probability)
        print('UCB score for action ',id_action,' : ',root_final.children[id_action].ucb_score(id_action))
        print('-------------------------')
