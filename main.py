import sys
import numpy as np
import math
import random
import gym
import gym_chess


def main_alpha_zero():
    env = gym.make('ChessAlphaZero-v0')
    env.reset()
    
    #Loop until the game is done
    done = False
    counter = 0
    while not done:
        print("Your Turn..... \n")
        
        action = random.sample(env.legal_actions,1)
        action = action[0]
        # Take the action and get the new state, reward, and whether the game is over
        state, reward, done, _ = env.step(action)
        print(env.render(mode='unicode'))
        
        counter += 1
    
    env.close()
    print("Reward: ", reward)
    print("Game Over in {} moves".format(counter))



def game():
    '''
    Simulation of a game between two agents AlphaZero and chess-v0
    '''
    env_alpha = gym.make('ChessAlphaZero-v0')
    env_v0 = gym.make('Chess-v0')
    
    env_alpha.reset()
    env_v0.reset()
    
    done_alpha = False
    done_v0 = False
    
    while not done_alpha and not done_v0:
        # AlphaZero's turn
        action_alpha = random.sample(env_alpha.legal_actions,1)
        action_alpha = action_alpha[0]
        state_alpha, reward_alpha, done_alpha, _ = env_alpha.step(action_alpha)
        
        # chess-v0's turn
        action_v0 = random.sample(env_v0.legal_actions,1)
        action_v0 = action_v0[0]
        state_v0, reward_v0, done_v0, _ = env_v0.step(action_v0)
        
        print(env_alpha.render(mode='unicode'))
        print(env_v0.render(mode='unicode'))
    
    
    
    
    
if __name__ == "__main__":
    #main()
    main_alpha_zero()