'''
Implement a Monte Carlo Tree Search algorithm for the game of chess.
'''
from os import cpu_count
from node_class import *


def compute_pi_posterior(root_node,temperature):
    '''
    Compute the posterior probability of the root node
        root_node: the root node of the tree
        temperature: the temperature parameter
    
    '''
    inv_temp = 1/temperature
    pi = torch.zeros(4672)
    actions_list = root_node.state.legal_actions()
    for id_action in range(len(actions_list)):
        pi[actions_list[id_action]] = root_node.children[id_action].visit_count#**inv_temp
        pi[actions_list[id_action]]= pi[actions_list[id_action]]/(root_node.visit_count)#**inv_temp)
    #Reset and delete the children nodes
    
    del root_node.children 
    return pi
  


    
def MCTS(root,num_simulations):
    '''
    Perform a Monte Carlo Tree Search algorithm for the game of chess
    root: the root node of the tree
    num_simulations: the number of simulations to perform
    actions_tracker: the list of actions that have been taken to reach the root node
    '''
    
    
    #First Expansion
    root.expand()
    cput = 1e4 # On choisit grand pour favoriser l'exploration et les outputs du réseaux de neurones
    
    for i in range(num_simulations):
        #print('Simulation number: ',i)
        if (i<=num_simulations*(2/3)):
            cput = cput*0.99
        else:
            cput = 1
        search_path = []
        search_path.append(root)
        
        #Select the child of the root node with the best ucb score
        current = root.select_child(cput)
        search_path.append(current)
        
        #Selection
        is_leaf = current.is_leaf
        while(not is_leaf):
            #browse the tree by selecting the best child
            
            current = current.select_child(cput)
            is_leaf = current.is_leaf 
            search_path.append(current)  
            
        #Expansion
     
        if (current.visit_count==0):
            if (not current.state.is_terminal()):
                prior_prob,value = current.neural_network.forward(current.obs) #rollout
                #Convert value to numpy float
                value = value.detach().numpy()[0][0]
                if (current.player_turn== root.player_turn):    #Backpropagation
                        update_path(search_path,value)
                else:
                    update_path(search_path,-value)
            
            else:
                value = current.state.player_reward(root.player_turn)
                update_path(search_path,value)
            
            
                
        else:
            
            
            #Simulation
            if (current.state.is_terminal()):
               
                value = current.state.player_reward(root.player_turn)
                #update_path(search_path,value) Don't need to update the path because the node is terminal and has already be updated
                
            
            else:
                current.expand()
                current = current.select_child(cput)
                search_path.append(current)
                
                if (not current.state.is_terminal()):
                    prior_prob,value = current.neural_network.forward(current.obs) #rollout
                    value = value.detach().numpy()[0][0]
                    if (current.player_turn== root.player_turn):    #Backpropagation
                        update_path(search_path,value)
                    else:
                        update_path(search_path,-value)
            
                else:
                    
                    value = current.state.player_reward(root.player_turn) #Vérifier si c'est le bon player turn mieux si on a id_winner
                    update_path(search_path,value)
                    
                
        #print('*********************************')
    temperature = 1
    #print('Loop done computing pi  ')
    pi =compute_pi_posterior(root,temperature)   
    
    return pi#,root




if __name__ == '__main__':
    
    neural_network = Alphazero_net()
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    obs_root = state.observation_tensor()
    player_turn = state.current_player()
    player_turn_root = player_turn
    parent =[]
    prob =1
    root = Node(state,obs_root,player_turn_root,parent,prob,player_turn,neural_network)
    '''
    pi,root = MCTS(root,1500)
    #print("Pi: ",pi)
    print('Root')
    root.summary()
    children = root.children
    count = 0
    for id_child in range (len(children)):
        print('Child'+str(id_child))
        children[id_child].summary()
    '''
        
    #Print the mean_action_value of the children nodes
    '''
    actions = root_final.state.legal_actions()
    for id_action in range(len(actions)):
        child = root_final.children[id_action]
        print('Action: ',actions)
        print('Pi: ',id_action,':',pi[actions[id_action]])
        print('Mean action value for action ',id_action,' : ',child.mean_action_value)
        print('Visit count for action ',id_action,' : ',child.visit_count)
        #print('Prior probability for action ',id_action,' : ',root_final.children[id_action].prior_probability)
        try:
            print(('UCB score for action ',id_action,' : ',root_final.ucb_score(id_action)))
        except:
            print('Problem with UCB score')
            print('Visit count',child.visit_count)
            print('Prior probability',child.prior_probability)
        print('-------------------------')
    
    print('Visit count: ',root_final.visit_count)
    '''