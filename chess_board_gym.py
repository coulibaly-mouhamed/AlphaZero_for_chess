import random 
import pyspiel
import numpy as np
#from open_spiel.python.algorithms.alpha_zero import alpha_zero
import torch

def naive_game():
    game = pyspiel.load_game("chess")
    state = game.new_initial_state()

    #clone_game = state.clone()
    counter = 0
    obs = state.observation_tensor()
    print('Observation Tensor shape : ',np.shape(obs))
    #converting to numpy array
    obs_np = np.array(obs)
    #Reshape obs to 8*8*20
    obs_np = obs_np.reshape(8,8,20)
    print('Observation Tensor shape after reshape : ',np.shape(obs_np))
    
    '''
    while not state.is_terminal():
        obs = state.observation_tensor()
        legal_actions = state.legal_actions()
        action = random.choice(legal_actions)
        state.apply_action(action)
        counter += 1
    '''
        #if counter ==5:
         #   clone_game = state.clone()
        #print('Current player: ',state.current_player())
        #clone_game.apply_action(action)
        #print(state)
        
    #print("Oberservation Shape: ", np.shape(obs))
    #Conver to torch tensor
    
    #obs_tensor = torch.Tensor(obs)
    #Reshape obs to 8*8*20
    #obs_tensor = obs_tensor.reshape(8,8,20)
    #print(obs_tensor)
    #print('Clone game: ',clone_game.legal_actions())
    
    #print(dir(state))
    #print(state.player_reward(1))
    #print(state.returns())
    '''
    print("Informartion Tensor")
    print(dir(state.information_state_tensor))
    print('--------------------------------')
    print("Observation Tensor Doc")
    print(state.observation_tensor.__doc__)
    print('--------------------------------')
    print("Observation Tensor")
    
    print("State")
    print(state)
    print('--------------------------------')
    '''
    
        
if __name__ == '__main__':
    naive_game()