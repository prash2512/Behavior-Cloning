import torch
from neural_net import NeuralNet
from load_policy import *
import gym
import torch.nn as nn

env = gym.make('CartPole-v1')
init = env.reset()
network = NeuralNet(4,128,64,2)
#network.load_state_dict(torch.load('mymodel.pt'))
_ , y = torch.max(network.forward(torch.Tensor(init).view(1,4)),0)[-1]
done = False
for i in range(1000):
    observation, reward, done, info = env.step(y.item())
    print(reward)
    if done==True:
        y = env.reset()
    observation = torch.Tensor(observation).view(1,4)
    env.render()
    _ , y = torch.max(network.forward(torch.Tensor(observation).view(1,4)),0)[-1]
