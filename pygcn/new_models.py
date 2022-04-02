import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution #20220112

from torch.nn import Parameter, Linear, Sequential, Module #20220112
import torch 
from utils import ReplayBuffer

import pdb

######################################################################
# model_name == 'GCN'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NN):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid) #20220122
        self.gc3 = GraphConvolution(nhid, nclass) #20220122 #如果gc3是output layer
        self.dropout = dropout
        self.NN = NN # Num of CBGs to be selected

    def apply_bn(self, x): #20220122 # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj):
        x = self.apply_bn(F.relu(self.gc1(x, adj))) #20220122 #先activate再BatchNorm
        x = self.apply_bn(F.relu(self.gc2(x, adj))) #20220122 #先activate再BatchNorm
        x = F.relu(self.gc3(x, adj)) #如果gc3是output layer
        return x


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


######################################################################
# model_name == 'MLP'

class MLPLayers(nn.Module): #20220120
    def __init__(self, nin, nhid1, nhid2, nout=1, activation='relu', bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        self.linear3 = Linear(nhid2, nout, bias=self.bias)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight) #20220122
        #torch.nn.init.kaiming_uniform_(self.linear2.weight) #20220122
        #torch.nn.init.kaiming_uniform_(self.linear3.weight) #20220122

    def apply_bn(self, x): #20220122 # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PoolLayer(nn.Module): #20220120
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        x = ((x.T)*(x[:,:,-1].T)).T #mask
        
        #output =  torch.flatten(x[:,torch.nonzero(x[0,:,-1]).squeeze(),:-1],start_dim=1,end_dim=-1) # 把NN个拿到疫苗的CBG的特征串联 #效果不好 #20220131 
        x_avg =  (torch.sum(x[:,:,:-1],axis=1)/(len(torch.nonzero(x[0,:,-1],as_tuple=True)[0]))) #初代模型
        #x_std = torch.std(x[:,(x[0,:,-1]).nonzero().squeeze(),:-1],dim=1) #20220131 #很多0，训练时loss变nan
        #x_max = torch.max(x[:,(x[0,:,-1]).nonzero().squeeze(),:-1],dim=1)[0] #20220131
        #x_min = torch.min(x[:,(x[0,:,-1]).nonzero().squeeze(),:-1],dim=1)[0] #20220131
        output = x_avg
        #output = torch.cat((x_avg, x_std,x_max, x_min), dim=1).squeeze()
        
        return output


######################################################################
# model_name == 'GNN_OVER_MLP'

class GCN_OVER_MLP(nn.Module): #20220121
    def __init__(self, config):
        nn.Module.__init__(self)
        self.GCNLayer = GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout,config.NN)
        self.PoolLayer = PoolLayer()
        self.MLPLayers = MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        self.dim_touched = config.dim_touched

    def forward(self, x, adj):
        #x = self.GCNLayer.forward(x, adj) 
        for i in range(x.shape[0]): #暂时无法批处理，只能土法循环  #20220121
            #this_output = self.GCNLayer.forward(x[i,:,:-1], adj) #最后一维标记是否免疫，不要动
            this_output = self.GCNLayer.forward(x[i,:,:self.dim_touched], adj) #20220123
            if(i==0):
                all_gcn_output = this_output.clone().unsqueeze(dim=0) #20220127
            else:
                all_gcn_output = torch.cat((all_gcn_output,this_output.unsqueeze(dim=0)), dim=0)
        all_gcn_output = torch.cat((all_gcn_output, x[:,:,self.dim_touched:]), dim=2) #20220123
        x = self.PoolLayer.forward(all_gcn_output)
        x = self.MLPLayers.forward(x)
        return x


######################################################################
# model_name == 'SoftGenerator'

class SoftGenerator(nn.Module): #20220203
    def __init__(self, config):
        nn.Module.__init__(self)
        self.GCN = SoftGeneratorGCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout,config.NN)
        #self.MLP = SoftGeneratorMLP(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        self.PoolMLP = SoftGeneratorPoolMLP(32, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_bias)
        self.Attention = SoftGeneratorAttention()
        
        self.dim_touched = config.dim_touched
        self.NN = config.NN #20220201
        self.saved_log_probs = [] #20220203
        self.rewards = [] #20220203
        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity) #20220205
        
    def forward(self, x, adj):
        all_gcn_output = self.GCN.forward(x[:,:self.dim_touched], adj) 
        #all_gcn_output = torch.cat((all_gcn_output, x[:,self.dim_touched:]), dim=1) #20220123 #20220206注释
        key = self.PoolMLP(all_gcn_output) #20220206
        attn = self.Attention(key, all_gcn_output)
        return attn


class SoftGeneratorGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NN):
        super(SoftGeneratorGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid) #20220122
        self.gc3 = GraphConvolution(nhid, nclass) #20220122 #如果gc3是output layer
        self.dropout = dropout
        self.NN = NN # Num of CBGs to be selected

    def apply_bn(self, x): #20220122 # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) #original #(20220121)leaky_relu也没用
        x = F.relu(self.gc2(x, adj)) #如果gc2是output layer
        x = F.relu(self.gc3(x, adj)) #如果gc3是output layer

        return x


class SoftGeneratorPoolMLP(nn.Module): #20220206
    def __init__(self,nin, nhid1, nhid2, nout=1, bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        #self.linear3 = Linear(nhid2, nout, bias=self.bias)
        self.linear3 = Linear(nhid2, nin, bias=self.bias)

    def forward(self, x): #对所有cbg的embedding取一个mean，过mlp，激活，得到key vector
        x = torch.mean(x,dim=0).unsqueeze(0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x # key vector


class SoftGeneratorAttention(nn.Module): #20220206
    def __init__(self):
        nn.Module.__init__(self)

    def apply_bn(self, x): #Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, key, x):
        attn = torch.mul(key,x).sum(dim=1) #torch.mul(key,x).sum(dim=1)[0]等同于torch.dot(key.squeeze(),x[0,:])
        attn = F.softmax(attn) #只有一个1
        #attn = (attn - attn.min()) / (attn.max()-attn.min()) #minmax scaling
        return attn


class SoftGeneratorMLP(nn.Module): #202201203
    def __init__(self, nin, nhid1, nhid2, nout=1, activation='relu', bias=True):
        nn.Module.__init__(self)
        self.bias = bias
        self.linear1 = Linear(nin, nhid1, bias=self.bias)
        self.linear2 = Linear(nhid1, nhid2, bias=self.bias)
        self.linear3 = Linear(nhid2, nout, bias=self.bias)

    def apply_bn(self, x): #20220122 #Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x):
        x = self.apply_bn(F.relu(self.linear1(x)))
        x = self.apply_bn(F.relu(self.linear2(x)))
        x = F.softmax(self.linear3(x),dim=0)
        return x


#######################################################################
# Get model

def get_model(config, model_name='GCN'):
    if(model_name=='GCN'):
        layers = Sequential(
            GCN(config.gcn_nfeat, config.gcn_nhid1, config.gcn_nhid2, config.gcn_nclass, config.gcn_dropout,config.NN), #20220119
            LinearLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    elif(model_name=='MLP'):
        layers = Sequential(
            PoolLayer(),
            MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    elif(model_name=='GNN_OVER_MLP'): #20220121
        layers = GCN_OVER_MLP(config)
    elif(model_name=='SoftGenerator'): #20220203
        layers = SoftGenerator(config)
    return layers

