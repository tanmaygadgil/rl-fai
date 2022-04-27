import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchsummary import summary
import sys
import os
import json
import random, string
import pandas as pd


env = gym.make('LunarLander-v2')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEM_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []
actual_rewards = []

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)
    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN():

    def __init__(self, network, gamma) -> None:
        self.memory = ReplayBuffer()
        self.epsilon = EXPLORATION_MAX
        self.network = network
        self.gamma = gamma

    def get_action(self, observation):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        
        state = torch.tensor(observation).float().detach()
        state = state.to(DEVICE)
        state = state.unsqueeze(0)
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def learn(self):
        if self.memory.mem_count < BATCH_SIZE:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        
        q_target = rewards + self.gamma * predicted_value_of_future * dones

        loss = self.network.loss(q_target, predicted_value_of_now)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.epsilon *= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)
    
    def returning_epsilon(self):
        return self.epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help = "Select Network Number", default=1, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0001, type=float)
    parser.add_argument("--episodes", help="number of episodes", default=200, type=int)
    parser.add_argument("--gamma", help="gamma", default=0.95, type=float)
    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict['network'] == 1:
        from network_1 import Network as Network
        network = Network(env.observation_space.shape, 
                        action_space, args_dict['lr'])
    elif args_dict['network'] == 2:
        from network_2 import Network as Network
        network = Network(env.observation_space.shape, 
                        action_space, args_dict['lr'])
    print(network)
    print(args_dict)
    agent = DQN(network, args_dict['gamma'])

    for i in range(1, args_dict['episodes']+1):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        score = 0

        while True:
            #env.render()
            action = agent.get_action(state)
            state_, reward, done, info = env.step(action)
            # env.render()
            state_ = np.reshape(state_, [1, observation_space])
            agent.memory.add(state, action, reward, state_, done)
            agent.learn()
            state = state_
            score += reward

            if done:
                if score > best_reward:
                    best_reward = score
                actual_rewards.append(score)
                average_reward += score 
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
                break
                
            episode_number.append(i)
            average_reward_number.append(average_reward/i)

    # plt.plot(episode_number, average_reward_number)
    # plt.show()

    exp_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(9))
    args_dict['algorithm'] = 'dqn'
    args_dict['environment'] = 'lunarlander-discrete'

    os.makedirs(f"./results/experiment_{exp_name}",)
    print(len(average_reward_number))
    print(len(actual_rewards))
    print(len(list(range(args_dict['episodes']))))
    with open(f"./results/experiment_{exp_name}/config.json", 'w') as file:
        json.dump(args_dict, file)

    df = pd.DataFrame({"episodes": list(range(args_dict['episodes'])), 
                       "rewards": actual_rewards})
    df.to_csv(f"./results/experiment_{exp_name}/rewards.csv", index=False)
    
    

