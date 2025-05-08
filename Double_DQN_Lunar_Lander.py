import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import time
from collections import deque


env = gym.make( "LunarLander-v3" ) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pre_existing_model = True

# Memory Buffer 
class ReplayBuffer:

    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, new_state, done ) :
        self.memory.append( (state, action, reward, new_state, done) ) 

    def sample(self, batch_size ):
        return random.sample( self.memory, batch_size ) 

    def __len__(self):
        return len(self.memory)

# DQN Model 
class DQN(nn.Module):

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear( in_dims, 256 ),
            nn.ReLU(),
            nn.Linear( 256, 256 ),
            nn.ReLU(),
            nn.Linear( 256, 128 ),
            nn.ReLU(),
            nn.Linear( 128, out_dims )
        )

    def forward(self,x):
        return self.network(x)

def manual_seed( seed = 42 ) :
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# defining states and actions
states = env.observation_space.shape[0]
actions = env.action_space.n

# Hyperparameters 
MAIN_BUFFER_MEMORY = 20000
MIN_BUFFER_MEMORY = 1000
BATCH_SIZE = 128
GAMMA = 0.995
TARGET_UPDATE = 6
REWARD_RANGE = 15
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0005

# initializing model and memory buffer 
buffer = ReplayBuffer( MAIN_BUFFER_MEMORY ) 
policy_net = DQN( states, actions ).to(device)
target_net = DQN( states, actions ).to(device)
target_net.load_state_dict( policy_net.state_dict() )
target_net.eval()

# optimizer and loss functions
optimizer = torch.optim.Adam( policy_net.parameters(), lr = LEARNING_RATE)
critetion = nn.MSELoss()


# Epsilon Greedy Action Selection
def action_selection(state, epsilon):
    if random.random() > epsilon:
        # exploitation
        state_tensor = torch.FloatTensor( state ).unsqueeze(0).to(device)
        with torch.no_grad():
            return policy_net( state_tensor ).max(1)[1].item()
    else:
        # exploration
        return env.action_space.sample()


def train( episodes = 400 ) :
    
    epsilon = EPSILON_START
    rewards_history = [ ] 
    
    for episode in range(episodes):
        
        total_reward = 0 
        done = False
        state,_ = env.reset()
    
        while not done : 
    
            action = action_selection( state, epsilon ) 
            new_state,reward,truncated,terminated,_ = env.step(action)
            done = truncated or terminated 
            reward = max(min( reward, REWARD_RANGE ), -REWARD_RANGE )# clipping reward to : (-10,10)
            buffer.add( state, action, reward, new_state, done)
            total_reward += reward 
            state = new_state
    
    
            if len(buffer) > MIN_BUFFER_MEMORY:
                data = buffer.sample( BATCH_SIZE ) 
                state_arr, action_arr, reward_arr, new_state_arr,  done_arr = zip(*data)
    
                state_tensors = torch.FloatTensor( np.array(state_arr) ).to(device) # 64 x 8 
                action_tensors = torch.LongTensor( np.array(action_arr) ).unsqueeze_(1).to(device) # 64 x 1
                new_state_tensors = torch.FloatTensor( np.array(new_state_arr) ).to(device) # 64 x 8 
                reward_tensors = torch.FloatTensor( np.array(reward_arr) ).unsqueeze_(1).to(device) # 64 x 1
                done_tensors = torch.FloatTensor( np.array(done_arr) ).unsqueeze_(1).to(device) # 64 x 1
    
                current_q_values = policy_net( state_tensors ).gather(1,action_tensors) # 64 x 8 : model (8 x 4 ) : 64 x 4 => 64 x 1 
    
                with torch.no_grad():
                    next_action_values = policy_net( new_state_tensors ).max(1)[1].unsqueeze_(1) # 64 x 1
                    #double DQN 
                    next_q_values = target_net( new_state_tensors ).gather(1,next_action_values) # 64 x 1        
    
                estimated_q_values = reward_tensors + GAMMA * next_q_values * ( 1 - done_tensors ) # 64 x 1 
    
                loss = critetion( estimated_q_values, current_q_values ) 
                
                optimizer.zero_grad()
                loss.backward()
                for i in policy_net.parameters():
                    i.grad.data.clamp_(-1,1)
                optimizer.step()
                
            if done :
                break

        rewards_history.append( total_reward ) 
        avg_reward = np.mean(rewards_history[-100:])

        # early stoopping 
        if avg_reward > 200 and len(rewards_history) == 100:
            print(f"Environment is solved in {episode+1} episodes! Average Reward : {avg_reward:.5f}")
            break
        
        if ( episode + 1 ) % TARGET_UPDATE == 0 :
            target_net.load_state_dict( policy_net.state_dict() ) 
        
        epsilon = max( EPSILON_END ,epsilon * EPSILON_DECAY ) 
    
        if (episode+1) % 10 == 0 : 
            print( f"Episode : {episode+1} / {episodes} Reward : {total_reward:.2f} Epsilon : {epsilon:.5f} Average Reward : {avg_reward:.5f} " ) 

    print("Training completed. Saving model...")
    torch.save( policy_net.state_dict(), "policy_net.pth" )

def evaluation( episodes = 10 ) :
    
    env_eval = env = gym.make( "LunarLander-v3", render_mode="human" )

    for episode in range(episodes):
        
        total_reward = 0 
        done = False
        state,_ = env_eval.reset()
    
        while not done : 
    
            env_eval.render()
            with torch.no_grad():
                state_tensor = torch.FloatTensor( state ).unsqueeze_(0).to(device)
                action = policy_net( state_tensor ).max(1)[1].item()
            state,reward,truncated,terminated,_ = env_eval.step(action)
            done = truncated or terminated 
            reward = max(min( reward, REWARD_RANGE ), -REWARD_RANGE )# clipping reward to : (-10,10)
            total_reward += reward 
            if done : 
                break
        
        print( f"Episode : {episode+1} / {episodes} Reward : {total_reward:.2f}" )             

try:
    # manual_seed( 42 )
    if not pre_existing_model:    
        start = time.time()
        train( episodes = 600 )
        train_time = time.time() - start
        print(f"Training time: {train_time:.2f} seconds or {train_time/60:.2f} minutes")

    else:
        try:
            policy_net.load_state_dict( torch.load( "policy_net.pth", weights_only=True, map_location= torch.device(device)) )
            print("Pre-existing model loaded.")
        except FileNotFoundError:
            print("Pre-existing model not found. Training from scratch.")
            train( episodes = 600 )

    evaluation( episodes = 10 )

except KeyboardInterrupt:
    print("Training interrupted.")

finally:
    
    env.close()
    print("Environment closed.")