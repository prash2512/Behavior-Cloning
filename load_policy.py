import pickle
import torch
import numpy as np
from torch.autograd import Variable

def load_policy(filename):
    with open(filename,'rb') as f:
        data = pickle.loads(f.read())
    return data

def get_batch():
    data  = load_policy('transitions.pkl')
    batchnum = 1
    states = list(map(lambda x:x[0],data[batchnum]))
    actions = list(map(lambda x:x[1],data[batchnum]))
    return Variable(torch.cat(states).view(-1,1,states[0].shape[1])),Variable(torch.cat(actions))
