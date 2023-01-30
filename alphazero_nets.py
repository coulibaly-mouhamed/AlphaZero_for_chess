'''
Implement the AlphaZero algorithm with PyTorch for the Chess game. 
'''

from mcts import *
from tqdm import tqdm 

'''
Create a convolutional block made of :
    -A convolution of 256 filters of kernel size 3x3 with stride 1 
    - batch normalization
    -A rectifier non linearity 
'''
import torch.nn.functional as F

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




class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(20, 256, 3, stride=1, padding=1) #change 22 par 21
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        # Convert to tensor
        #s = torch.tensor(s, dtype=torch.float32) 
        s = s.view(-1, 20,8,8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

'''
Create a residual block made of :
    -A convolution of 2 filters of kernel size 1*1 with stride 1
    - batch normalization
    - A skip connection that adds the input to the block
    -A rectifier non linearity
'''
class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
        
class Alphazero_net(nn.Module):
    '''
    Implement the AlphaZero algorithm with PyTorch for the Chess game. It consists in a 
    Residual Network with 19 layers and 256 filters.
    '''
    def __init__(self):
        super(Alphazero_net, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            self.add_module('resblock'+str(block),ResBlock())
        self.outblock = OutBlock()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.1)
        
    def forward(self, x):
        
        s = self.conv(x)
        for block in range(19):
            s = self.__getattr__('resblock'+str(block))(s)
        p,v = self.outblock(s)
        return p,v
    
    def checkpoint(self,epoch):
        torch.save(self.state_dict(), 'checkpoint_'+str(epoch)+'.pth.tar')
        print('Checkpoint saved !')
        
    def loss_function(self,p,v,pi,z):
        '''
        Compute the loss function of the AlphaZero algorithm which is the sum of the 
        cross entropy loss and the MSE loss.
        
        '''
        return -torch.sum(pi*torch.log(p)) + torch.sum((z-v)**2)
    
     
    def fit_to_self_play(self,data_batch,cpu=1):
        '''
        Training Pipeline of the AlphaZero algorithm which consists in :
            - From data generated by self play do
            - Predict the output of the neural network
            - Compute the loss function
            - Backpropagate the loss function
            - Update the weights of the neural network
        '''
        running_loss = 0.0
        loss_batch = 0
        state_batch, pi_batch, v_batch = data_batch['state'], data_batch['pi'], data_batch['reward']
        #Convert to tensor state_batch, pi_batch, v_batch
        
        #for i in range(len(state_batch)):
        #    state_batch[i] = torch.from_numpy(state_batch[i])
        #    pi_batch[i] = torch.from_numpy(pi_batch[i])
        #    v_batch[i] = torch.tensor(v_batch[i])
        pi_batch = torch.from_numpy(np.array(pi_batch))
        state_batch = torch.from_numpy(np.array(state_batch))
        v_batch = torch.tensor(v_batch)
        
        state_batch = state_batch.float()
        pi_batch = pi_batch.float()
        v_batch = v_batch.float()
        
        self.optimizer.zero_grad()
        p_predicted, v_predicted = self.forward(state_batch)
        #print type (p_predicted)
        loss_batch = self.loss_function(p_predicted, v_predicted, pi_batch, v_batch)
        loss_batch.backward()
        self.optimizer.step()
        running_loss += loss_batch.item()
        
        return loss_batch
    
    
    def evaluator (self):
        pass
    
    
    def self_play(self,n_simu):
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
        neural_network = self
        #neural_network = neural_network.to('cuda')
        game = pyspiel.load_game("chess")
        state = game.new_initial_state()
        obs_root = state.observation_tensor()
        player_turn = state.current_player()
        player_turn_root = player_turn
        parent =[]
        prob =1
        root = Node(state,obs_root,player_turn_root,parent,prob,player_turn,neural_network)
    
        play =0
        is_terminal = root.state.is_terminal()
        #Convert the observation root list to numpy array
        obs_root_numpy = np.array(obs_root)
        #reshape to 8*8*20
        obs_root_numpy = np.reshape(obs_root_numpy,(8,8,20))
        state_history.append(obs_root_numpy)
        while not is_terminal:
            
                
            #Run MCTS
            #print('Run MCTS')
            pi = MCTS(root,n_simu)
            #print('End MCTS')
            pi_numpy = pi.detach().numpy()
            pi_history.append(pi_numpy)
            #Select the action
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
            obs_root_numpy = np.array(obs_root)
            obs_root_numpy = np.reshape(obs_root_numpy,(8,8,20))
            state_history.append(obs_root_numpy)
            root = Node(next_root_state,obs_root,player_turn_root,parent,prob,player_turn,neural_network)
            
            is_terminal = root.state.is_terminal()
        
            
            play += 1
            #print('___________________________________________________________________')
            
        state_history = np.array(state_history).reshape(-1,8,8,20)
        pi_history = np.array(pi_history).reshape(-1,4672)
        reward = root.state.player_reward(player_turn_root)
        print("play = ",play)
       
        return state_history,pi_history,reward

    
    
    def update_parameters(self,data,batch_size=500):  
        data_state,pi_games,reward_games = data['state'],data['pi'],data['reward']
       
        id_list = np.random.randint(0,len(data_state),batch_size)
        data_batch_state = data_state[id_list]
        pi_games_batch = pi_games[id_list]
        reward_games_batch = reward_games[id_list]
        data_batch ={'state':data_batch_state,'pi':pi_games_batch,'reward':reward_games_batch}
        print('Optim')
        loss_batch = self.fit_to_self_play(data_batch)
    
    
    def train(self,epochs=1000):
        '''
        Training Pipeline of the AlphaZero algorithm which consists in :
            - Generate data from self play games monitored by an MCTS
            - Update the parameters of the neural network
        '''
        
        
        for i in tqdm(range(epochs)):
            data_state = None
            pi_games = None
            reward_games = None
            n_games = 1
            #Generate data from self play games
            print('Begin Self play')
            for j in range(n_games):
                
                state_games,pi_game,reward = self.self_play(3)
                if (j==0):
                    data_state = state_games
                    pi_games = pi_game
                    reward_games = np.repeat(reward,len(pi_game))
                else:
                    #Concatenate data_state with state_games
                    data_state = np.concatenate((data_state,state_games),axis=0)
                    pi_games = np.concatenate((pi_games,pi_game),axis=0)
                    reward_games = np.concatenate((reward_games,np.repeat(reward,len(pi_game))),axis=0)
                
              
            print('End Self play')
            #Create a list with reward for each state
            
            data={'state':data_state,'pi':pi_games,'reward':reward_games}
            print('Updating parameters')
            
            self.update_parameters(data,50)
            
            if i%50 == 0:
                self.checkpoint(i)
        self.checkpoint(epochs)    
        
        

if __name__ == '__main__':
    neural_network = Alphazero_net()
    print('Training')
    neural_network.train(50)
    print('End Training')
    