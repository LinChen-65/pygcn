from multiprocessing import Pool
import torch.nn as nn
import torch.nn.functional as F
#from pygcn.layers import GraphConvolution #original
from layers import GraphConvolution #20220112

from torch.nn import Parameter, Linear, Sequential, Module #20220112
from activaition import get as get_activation
import torch 

import pdb

class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2) #20220119
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


class LinearLayers(nn.Module): #20220112
    def __init__(self, nin, nhid1, nhid2, nout=1, activation="relu", bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        self.linear3 = Linear(nhid2, nout, bias=self.bias)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        

class MLPLayers(nn.Module): #20220120
    def __init__(self, nin, nhid1, nhid2, nout=1, activation="relu", bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        self.linear3 = Linear(nhid2, nout, bias=self.bias)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PoolLayer(nn.Module): #20220120
    def __init__(self, nin, nout, activation="relu", bias=True):
        nn.Module.__init__(self)

    def forward(self, x):
        #pdb.set_trace()
        x = ((x.T)*(x[:,:,-1].T)).T #mask
        x_avg =  (torch.sum(x[:,:,:-1],axis=1)/(len(torch.nonzero(x[0,:,-1],as_tuple=True)[0])))
        #x_std = 
        #x_avg =  torch.mean(x[torch.nonzero(x[:,:,-1],as_tuple=True)[0],torch.nonzero(x[:,:,-1],as_tuple=True)[1],:-1], axis=0).unsqueeze(axis=1)
        #x_std =  torch.std(x[torch.nonzero(x[:,:,-1],as_tuple=True)[0],torch.nonzero(x[:,:,-1],as_tuple=True)[1],:-1], axis=0).unsqueeze(axis=1)
        
        #x = x * (x[:,-1].unsqueeze(axis=1)) #mask
        #x_avg =  torch.mean(x[torch.nonzero(x[:,-1],as_tuple=True)[0],:-1], axis=0).unsqueeze(axis=1)
        #x_std =  torch.std(x[torch.nonzero(x[:,-1],as_tuple=True)[0],:-1], axis=0).unsqueeze(axis=1)
        #output = torch.cat((x_avg, x_std), dim=0).squeeze()
        output = x_avg
        return output

def get_model(config, model_name='GCN'):
    if(model_name=='GCN'):
        layers = Sequential(
            #GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout), #original
            GCN(config.gcn_nfeat, config.gcn_nhid1, config.gcn_nhid2, config.gcn_nclass, config.gcn_dropout), #20220119
            LinearLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    elif(model_name=='MLP'):
        layers = Sequential(
            PoolLayer(config.pool_nin, config.pool_nout),
            MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    return layers