'''
    In this script we will implement the self play method based on MCTS algorithm guided by a neural network
    
'''

from mcts import *

def choose_action(pi,env):
    '''
        Choose an action according to the probability distribution pi
        Input:
            -pi: probability distribution
        Output:
            -id_action: the id of the chosen action
    '''
    id_action = 5000
    while (id_action not in env.legal_actions):
        id_action = np.random.choice(len(pi),p=pi)
    return id_action


def play_mcts_guided_game(neural_network,n_simu):
    '''
        Simulation of a single game with MCTS guided by a neural network 
        Inputs:
            -neural_network 
        Ouputs: 
            - Data_frame with for each state of the game the vector of probability pi 
    '''
    state_history = []
    pi_history = []
    
    #Initialsation of the environnement
    env = gym.make('Chess-v0')
    env = BoardEncoding(env,history_length=1)
    env = MoveEncoding(env)
    board = env.reset()
    
    #Initialization of state_history and pi_history
    state_history.append(board)
    
    
    done = False 
    
    #Initialisation of the root node
    player_turn = 1
    root = Node(env,board,neural_network,1,1)
    while (not done):
        #Run MCTS
        root_final,pi = MCTS(root,n_simu)
        pi_history.append(pi)
        #Select the action
        id_action = choose_action(pi,env)
        #Update the environnement and play the action
        new_state, reward, done, info = env.step(id_action)
        player_turn *= -1
        root = Node(env,new_state,neural_network,player_turn,1)
        #Save the state and the probability vector
        state_history.append(new_state)
    return state_history,pi_history,-reward*player_turn
        
if __name__ == '__main__':
    neural_network = Alphazero_net()
    state_history,pi_history,reward=play_mcts_guided_game(neural_network,100)
    