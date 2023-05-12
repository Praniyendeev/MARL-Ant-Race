# Step 1: Import required libraries and create the simple tag environment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import gym
import os
import time
from PIL import Image
import cv2
import json
# Create the simple tag environment
env = gym.make('Race-v0')#simple_tag_v2.parallel_env(num_obstacles = 0, max_cycles=50, render_mode="human")
obs = env.reset()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.state1 = nn.Linear(state_dim, hidden_dim)
        self.state2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.action1 = nn.Linear(action_dim, hidden_dim // 2)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state, action):
        s1 = F.leaky_relu(self.state1(state))
        s2 = F.leaky_relu(self.state2(s1))
        a1 = F.leaky_relu(self.action1(action))

        x = torch.cat((s2, a1), dim=1)
        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)
        return x

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, agent, device,name=""):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        #print(state_dim, action_dim, hidden_dim,"val")
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.name = str(agent)+name
        
        self.device = device
        self.update_target_networks()
        

    def update_target_networks(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state, noise=None,eps=0.1):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state).cpu().data.numpy()

        if noise!=None:
            if np.random.rand() < eps:
                action = noise.noise()*1
            else:
                action += noise.noise()*0.3
            return action

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)



def train_maddpg(agents, env, episodes, noise, device, batch_size=128, gamma=0.99, tau=0.05,nstate=10,naction=10,add=False,run_name="",mr=np.array([168,168])):

    memory = deque(maxlen=100000)
    rewards_list = []
    max_reward=168
    max_rewards=mr#np.array([-100,-100])
    epsbeg = 0.1
    epsend = 0.05
    epsdecay = 1e-4

    tstep=0
    for episode in range(episodes):
        states = env.reset()
        episode_reward = 0
        episode_rewards =np.array([0.0,0.0])
        start=time.time()
        ep_len=0
        
        
        eps_frame=[None]*1000
        frm_ct=0
        while True:
            tstep+=1
            
            epsilon = epsend + (epsbeg - epsend) * np.exp(-1. * tstep *epsdecay) # exponential decay

            actions = np.concatenate([agent.act(states[:], noise,eps=epsilon) for i,agent in enumerate(agents)])
            
            if add:
                z=np.zeros_like(actions)
                actions=np.concatenate([actions,z])
            action_vals = np.argmax(actions)
            #print(actions) 
            k =env.step(actions)
            next_states, rewards, dones, _= k
            #env.render()
            if ep_len%10==0:
                screen = env.render(mode='rgb_array')
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                eps_frame[frm_ct]=screen
                frm_ct+=1
            ep_len+=1


            if type(dones) is not list:
                dones=[dones]*len(agents)
            else:

                dones=[dones[i] for i,a in enumerate(agents)]

            memory.append((states, actions, rewards, next_states, dones))
            episode_reward += np.sum(rewards)
            episode_rewards+=np.array(rewards)
            if type(dones) is list:
                if all(done == True for done in dones):
                    break
            elif dones == True:  
                break
        
            if len(memory) >= batch_size: 
                experiences = random.sample(memory, batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*experiences)
                    
                # Extracting and organizing data from memory.
                actors_states = {}
                actors_actions = {}
                actors_rewards = {}
                actors_next_states = {}
                actors_dones = {}

                for agent,_ in enumerate(agents):
                    actors_states[agent] = torch.stack([torch.from_numpy(batch_states[itr][:]).float().to(device) for itr in range(batch_size)])
                    actors_actions[agent] = torch.stack([torch.from_numpy(batch_actions[itr][naction*agent:naction*(agent+1)]).float().to(device) for itr in range(batch_size)])
                    actors_rewards[agent] = torch.Tensor([batch_rewards[itr][agent] for itr in range(batch_size)]).to(device)
                    actors_next_states[agent] = torch.stack([torch.from_numpy(batch_next_states[itr][:]).float().to(device) for itr in range(batch_size)])
                    actors_dones[agent] = torch.Tensor([batch_dones[itr][agent]*1 for itr in range(batch_size)]).to(device)
        
                batch_states = torch.cat([actors_states[agent] for agent in range(len(agents))], dim = 1)
                batch_next_states = torch.cat([actors_next_states[agent] for agent in range(len(agents))], dim = 1)
                batch_actions = torch.cat([actors_actions[agent] for agent in range(len(agents))], dim = 1)
        
                # Preparing data for Critic Model
#                 batch_states = torch.from_numpy(batch_states).float().to(device)
#                 batch_actions = torch.from_numpy(batch_actions).float().to(device)
#                 batch_next_states = torch.from_numpy(batch_next_states).float().to(device)
                
#                 batch_dones = torch.from_numpy(np.array(batch_dones).astype(np.uint8)).float().to(device)

                for i,agent in enumerate(agents):
                    # Update the critic

                    next_actions = agent.target_actor(actors_next_states[i])
                    #next_actions = torch.cat(next_actions, dim=1)

                    target_q_values = agent.target_critic(batch_next_states[:,nstate*i:nstate*(i+1)], next_actions)
                    expected_q_values = actors_rewards[i][:,None] + (1 - actors_dones[i][:,None]) * gamma * target_q_values.to(device)
                    q_values = agent.critic(batch_states[:,nstate*i:nstate*(i+1)], batch_actions[:,i*naction:(i+1)*naction])
                    critic_loss = F.mse_loss(q_values, expected_q_values.detach())
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic_optimizer.step()

                    # Update the actor
                    actions = agent.actor(actors_states[i])
                    #actions = torch.cat(actions, dim=1)
                    actor_loss = -agent.critic(batch_states[:,nstate*i:nstate*(i+1)], actions).mean()
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                    # Update target networks
                    soft_update(agent.target_actor, agent.actor, tau)
                    soft_update(agent.target_critic, agent.critic, tau)
                    
            states = next_states
        saved=[""]*len(agents)
        avgrew=episode_rewards/ep_len


        

        for i,agent in enumerate(agents):
            if avgrew[i]>max_rewards[i]:
                max_rewards[i] =avgrew[i]
                saved[i]="S"
                torch.save(agent.actor.state_dict(), f"models/{agent.name}_actor.pth")
                torch.save(agent.critic.state_dict(), f"models/{agent.name}_critic.pth")

                np.save(f"models/{agent.name}_mr.npy",max_rewards)

        rewards_list.append(episode_rewards)
        if "S" in saved or episode%50==0:
            print("video Saved")
            writer = cv2.VideoWriter('videos/{}_{}.mp4'.format(episode,run_name), cv2.VideoWriter_fourcc(*'mp4v'), 15, (500,500))
            for frame in eps_frame:
                if np.all(frame ==None):
                  break
                writer.write(frame)
            writer.release()
            np.save(f"models/{run_name}_reward.npy",np.array(rewards_list))

        
        # if episode_rewards> max_reward:
        #     max_reward =episode_rewards
        #     saved="Saved"
        #     for agent in agents:
        #         torch.save(agent.actor.state_dict(), f"models/{agent.name}_actor.pth")
        #         torch.save(agent.critic.state_dict(), f"models/{agent.name}_critic.pth")

        
        
        time_epsiode=(time.time()-start)/60
        avgrew=np.round(avgrew,0)
        print(f"Episode {episode + 1}/{episodes}, Reward: {avgrew} , {time_epsiode:.3f}_{ep_len}, {saved}")
        image_array=env.render(mode='rgb_array')
        output_image = Image.fromarray(image_array)
        directory="images/"+run_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        output_image.save(directory+"/{}_{}.jpg".format(episode,np.round(avgrew,0)))

    return rewards_list


def test_maddpg(agents, env, episodes):
    for episode in range(episodes):
        states = env.reset()
        episode_rewards = 0

        while True:
            actions = {agent.name : np.argmax(agent.act(states[agent.name])) for agent in agents}
            next_states, rewards, _, dones, _ = env.step(actions)
            env.render()
            episode_rewards += np.sum(list(rewards.values()))

            if all(value == True for value in dones.values()):
                break

            states = next_states

        print(f"Test Episode {episode + 1}/{episodes}, Reward: {episode_rewards}")



class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


# if __name__ == '__main__':

    
def run(name="twoA"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_agents = 1#len(env.agents)
    state_dims = env.observation_space.shape[0]#{agent : env.observation_space(agent).shape[0] for agent in env.agents}
    action_dim = env.action_space.shape[0]#('agent_0').n

    print(state_dims, action_dim, '\n')#type(env.action_space('adversary_0')))

    hidden_dim = 512
    lr_actor = 1e-3
    lr_critic = 1e-3
    episodes = 2000
    test_episodes = 10
    
    add=False

    nstate=state_dims#//n_agents
    naction=action_dim//n_agents
    # Create the agents
    agents = [MADDPGAgent(nstate, naction, hidden_dim, lr_actor, lr_critic, agent, device,name=name) for agent in range(n_agents)]
    print(len(agents))
    # Create noise for exploration
    noise = OUNoise(naction)
    reward_init=10
    mr=np.array([reward_init,reward_init])

    for agent in agents:
        actor_path = f"models/{agent.name}_actor.pth"
        critic_path = f"models/{agent.name}_critic.pth"
        mr_path = f"models/{agent.name}_mr.npy"

        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path))
        else:
            print(f"Actor model not found for {agent.name} agent")
            
        if os.path.exists(critic_path):
            agent.critic.load_state_dict(torch.load(critic_path))
        else:
            print(f"Critic model not found for {agent.name} agent")

        if os.path.exists(mr_path):
            mr =np.load(mr_path)


    # Train the agents
    rewards = train_maddpg(agents, env, episodes, noise, device,nstate=nstate,naction=naction,add=add,run_name=name,mr=mr)

    for agent in agents:
        torch.save(agent.actor.state_dict(), f"models/{agent.name}_actor.pth")
        torch.save(agent.critic.state_dict(), f"models/{agent.name}_critic.pth")

    for agent in agents:
            agent.actor.load_state_dict(torch.load(f"models/{agent.name}_actor.pth"))
        
    # Test the trained agents
    test_maddpg(agents, env, test_episodes)

    # Save Rewards
    np.save(f"simple_tag_models/rewards.npy", np.array(rewards))

