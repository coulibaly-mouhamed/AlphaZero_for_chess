from email import policy
import gym 
import gym_chess
import random 
import chess
from gym_chess.alphazero import BoardEncoding
from gym_chess.alphazero import MoveEncoding
from alphazero_nets import *
import numpy as np 
'''
In this script we will create environnement for the chess game 
'''

def random_model_choice(board,alphazero):
    policy, value = alphazero.forward(board)
    #Return the id of the action with the highest policy
    # Choose a random id from the list of ids with the highest policy
    id_2 = random.choice(np.argwhere(policy == np.amax(policy)).flatten())
    #id_action = np.argmax(policy.detach().numpy())
    return id_2



class game():
    
    def __init__(self,env,player_turn):
        self.env = env
        self.player_turn = player_turn # 1 for white and -1 for black
        self.board = env.reset()
        self.neural_net = Alphazero_net()
        
    def play(self,n_coups):
        done = False
        for i in range(n_coups):
            if self.player_turn == 1:
                action = random.choice(self.env.legal_actions)
                self.board, reward, done, info = self.env.step(action)
            else:
                found_action = False
                count = 0
                while (found_action and count < 100000):
                    action = random_model_choice(self.board,self.neural_net)
                    try:
                        self.board.push(self.env.action_to_move(action))
                        self.board, reward, done, info = self.env.step(action)
                        count += 1
                        found_action = True
                    except:
                        count += 1
                        print("count = ",count)   
                
            self.player_turn *= -1
            
            if done:
                print("Game Over")
                print("Reward", reward)
                print('Winner is : ',-self.player_turn)
                print(self.env.render(mode='unicode'))
                break
            
        return self.env,self.board #,self.board, reward, done, info
    
if __name__ == '__main__':
    env = gym.make('Chess-v0')
    env = BoardEncoding(env,history_length=1)
    env = MoveEncoding(env)
    env.reset()
    game = game(env,1)
    game.play(50)
    print(env.render(mode='unicode'))
    