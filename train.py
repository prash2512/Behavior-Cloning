import torch
from neural_net import NeuralNet
from load_policy import *
import gym
import torch.nn as nn
import torch.optim as optim

network = NeuralNet(4,128,64,2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(),0.01,0.9)

losses = []
iter = []
for i in range(20):
    state,action = get_batch()
    for j in range(state.size()[0]):
        network.zero_grad()
        output = network(state[j])
        loss = criterion(output,action[j])
        if i%10==0:
            losses.append(loss)
            iter.append(i)
        loss.backward()
        optimizer.step()
    print(loss)

torch.save(network.state_dict(),"mymodel.pt")
