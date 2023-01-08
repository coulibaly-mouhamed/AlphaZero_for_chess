'''
    In this script we will implement the self play method based on MCTS algorithm guided by a neural network
    
'''

from mcts import *
import os 
directory = os.getcwd()

def choose_action(pi,state):
    '''
        Choose an action according to the probability distribution pi
        Input:
            -pi: probability distribution
        Output:
            -id_action: the id of the chosen action
    '''
    id_action = 5000
    try_ = 0
    #convert pi to numpy array
    pi_np = pi.detach().numpy()
    actions  = state.legal_actions()
    while (id_action not in actions):
    
        id_action = np.random.choice(len(pi),p=pi_np)
        try_ += 1
    #print('Find action after ',try_,' tries')
    return id_action


def play_mcts_guided_game(neural_network,n_simu,num_game):
    '''
        Simulation of a single game with MCTS guided by a neural network 
        Inputs:
            -neural_network 
        Ouputs: 
            - Data_frame with for each state of the game the vector of probability pi 
    '''
    state_history = []
    pi_history = []
    
    #Initialization of state_history and pi_history
    #state_history.append(obs_init)
   
    #Initialisation of the root node
    neural_network = Alphazero_net()
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()
    obs_root = state.observation_tensor()
    player_turn = state.current_player()
    player_turn_root = player_turn
    parent =[]
    prob =1
    root = Node(state,obs_root,player_turn_root,parent,prob,player_turn,neural_network)
    #print('Node attributes',dir(root))
    #print('Expansion')
    #root.expand()
    play =0
    is_terminal = root.state.is_terminal()
    
    state_history.append(obs_root)
    while not is_terminal:
        print("play = ",play)
        #Run MCTS
        #print('Run MCTS')
        pi = MCTS(root,n_simu)
        #print('End MCTS')
        
        pi_history.append(pi)
        #Select the action
        '''
        if  (pi.sum() != 1):
            print("pi.sum() = ",pi.sum())
            print('Problem with pi')
            #Print pi 
            break
        '''
        id_action = choose_action(pi,root.state)
        #Update the environnement and play the action
        root.state.apply_action(id_action)
        
        next_root_state = root.state.clone()
        if (next_root_state.is_terminal()):
            print('Game over')
            break
        player_turn = next_root_state.current_player()
        parent = []
        obs_root = next_root_state.observation_tensor()
        root = Node(next_root_state,obs_root,player_turn_root,parent,prob,player_turn,neural_network)
        
        is_terminal = root.state.is_terminal()
        #Save the state and the probability vector
        #print('Legal actions: ',root.state.legal_actions())
        
        play += 1
        #print('___________________________________________________________________')
        
         
    reward = root.state.player_reward(player_turn_root)
    
    #Save the neural network weights, the state history and the pi history in the folder data_self_play
         #Go to the folder data_self_play
    os.chdir(directory+'/data_self_play')
        #Create a folder for the game
    os.mkdir('game_'+str(num_game))
        #Go to the folder game_num_game
    os.chdir(directory+'/data_self_play/game_'+str(num_game))
    #Save the neural network weights
    torch.save(neural_network.state_dict(), 'neural_network_weights'+str(num_game)+'.pt')
    #Save the state history and the pi history and the reward
     #Convert the state history to numpy array
    state_history_np = np.array(state_history)
     #Convert the pi history to numpy array
    pi_history_np = np.array(pi_history)
    #Save the state history and the pi history
    np.save('state_history'+str(num_game)+'.npy',state_history_np)
    np.save('pi_history'+str(num_game)+'.npy',pi_history_np)
    np.save('reward'+str(num_game)+'.npy',reward)
    
    os.chdir(directory)
    
    return #state_history,pi_history,reward
        
if __name__ == '__main__':
    neural_network = Alphazero_net()
    for game in range (10):
        print('Game number: ',game)
        play_mcts_guided_game(neural_network,5,game)
        print('-------------------------------------------------------------------')
    #print('Reward: ',reward)
    print('End of the games')