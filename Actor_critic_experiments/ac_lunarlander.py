  
import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
import pandas as pd
import argparse
import random, string
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("LunarLander-v2").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters, gamma, lr, save=False):
    all_rewards = []
    optimizerA = optim.Adam(actor.parameters(), lr = lr)
    optimizerC = optim.Adam(critic.parameters(), lr = lr)
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()
        score = 0

        for i in count():
#             env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()
            score += reward
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                # print('Iteration: {}, Score: {}'.format(iter, score))
                all_rewards.append(score)
                break
            if i > 5000:
                all_rewards.append(score)
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    if save:    
        torch.save(actor, 'model/actor.pkl')
        torch.save(critic, 'model/critic.pkl')
    env.close()
    return all_rewards


if __name__ == '__main__':
    # if os.path.exists('model/actor.pkl'):
    #     actor = torch.load('model/actor.pkl')
    #     print('Actor Model loaded')
    # else:
    #     actor = Actor(state_size, action_size).to(device)
    # if os.path.exists('model/critic.pkl'):
    #     critic = torch.load('model/critic.pkl')
    #     print('Critic Model loaded')
    # else:
    #     critic = Critic(state_size, action_size).to(device)
    start=datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", help = "Select Network Number", default=11, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0001, type=float)
    parser.add_argument("--episodes", help="number of episodes", default=200, type=int)
    parser.add_argument("--gamma", help="gamma", default=0.95, type=float)
    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict['network'] == 11:
        from network_1 import Network as Network
        actor = Network(env.observation_space.shape[0], action_size, is_actor=True).to(device)
        critic = Network(env.observation_space.shape[0], action_size).to(device)
    elif args_dict['network'] == 12:
        from network_2 import Network as Network2
        from network_1 import Network as Network1
        actor = Network1(env.observation_space.shape[0], action_size, is_actor=True).to(device)
        critic = Network2(env.observation_space.shape[0], action_size).to(device)
        
    elif args_dict['network'] == 21:
        from network_2 import Network as Network2
        from network_1 import Network as Network1
        actor = Network2(env.observation_space.shape[0], action_size, is_actor=True).to(device)
        critic = Network1(env.observation_space.shape[0], action_size).to(device)
        
    elif args_dict['network'] == 22:
        from network_2 import Network as Network
        actor = Network(env.observation_space.shape[0], action_size, is_actor=True).to(device)
        critic = Network(env.observation_space.shape[0], action_size).to(device)
    elif args_dict['network'] == 33:
        from network_3 import Network as Network
        actor = Network(env.observation_space.shape[0], action_size, is_actor=True).to(device)
        critic = Network(env.observation_space.shape[0], action_size).to(device)


    all_rewards = trainIters(actor, critic, n_iters=args_dict['episodes'],  lr=args_dict['lr'], gamma=args_dict['gamma'], save=False)
    exp_name = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(9))
    args_dict['algorithm'] = 'ac'
    args_dict['environment'] = 'lunarlander'
<<<<<<< HEAD
    plt.plot(list(range(args_dict['episodes'])), all_rewards)
    plt.show()
=======
    # plt.plot(list(range(args_dict['episodes'])), all_rewards)
    # plt.show()
>>>>>>> 47ca8a3141eca6124f6655a50c511d9b3159f82b

    os.makedirs(f"./results/experiment_{exp_name}",)
    with open(f"./results/experiment_{exp_name}/config.json", 'w') as file:
        json.dump(args_dict, file)

    df = pd.DataFrame({"episodes": list(range(args_dict['episodes'])), 
                       "rewards": all_rewards})
    df.to_csv(f"./results/experiment_{exp_name}/rewards.csv", index=False)
<<<<<<< HEAD
=======
    print(f"time taken: {datetime.now() -  start}")
>>>>>>> 47ca8a3141eca6124f6655a50c511d9b3159f82b
