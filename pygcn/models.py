from multiprocessing import Pool
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
#from pygcn.layers import GraphConvolution #original
from layers import GraphConvolution #20220112

from torch.nn import Parameter, Linear, Sequential, Module #20220112
from activaition import get as get_activation
import torch 
from torch_geometric.nn import TopKPooling #20220127

import pdb

NN = 70 # Num of CBGs to be selected

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        #self.gc2 = GraphConvolution(nhid, nclass) #如果gc2是output layer
        self.gc2 = GraphConvolution(nhid, nhid) #20220122

        #self.gc3 = GraphConvolution(nhid, nclass) #20220122 #如果gc3是output layer
        self.gc3 = GraphConvolution(nhid, nhid) #20220122

        #self.gc4 = GraphConvolution(nhid, nclass) #20220122 #如果gc4是output layer
        self.gc4 = GraphConvolution(nhid, nhid) #20220122

        self.gc5 = GraphConvolution(nhid, nclass) #20220122 #如果gc5是output layer
        #self.gc5 = GraphConvolution(nhid, nhid) #20220122

        #self.gc6 = GraphConvolution(nhid, nclass) #20220122 #如果gc6是output layer

        self.dropout = dropout

    def apply_bn(self, x): #20220122 #BatchNorm
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj):
        #x = F.relu(self.gc1(x, adj)) #original #(20220121)leaky_relu也没用
        #x = self.apply_bn(x) #20220122 #BatchNorm
        #x = F.relu(self.apply_bn(self.gc1(x, adj))) #20220122 #先BatchNorm再activate
        x = self.apply_bn(F.relu(self.gc1(x, adj))) #20220122 #先activate再BatchNorm
        #x = F.dropout(x, self.dropout, training=self.training)

        #x = self.gc2(x, adj) #如果gc2是output layer
        #x = F.relu(self.gc2(x, adj))
        #x = self.apply_bn(x) #20220122 #BatchNorm
        #x = F.relu(self.apply_bn(self.gc2(x, adj))) #20220122 #先BatchNorm再activate
        x = self.apply_bn(F.relu(self.gc2(x, adj))) #20220122 #先activate再BatchNorm
        #x = F.dropout(x, self.dropout, training=self.training)

        #x = self.gc3(x, adj) #如果gc3是output layer
        #x = F.relu(self.gc3(x, adj)) #如果gc3是output layer
        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.relu(self.apply_bn(self.gc3(x, adj))) #20220122 #先BatchNorm再activate
        x = self.apply_bn(F.relu(self.gc3(x, adj))) #20220122 #先activate再BatchNorm

        #x = F.relu(self.gc4(x, adj)) #如果gc4是output layer
        #x = F.relu(self.apply_bn(self.gc4(x, adj))) #20220122 #先BatchNorm再activate
        x = self.apply_bn(F.relu(self.gc4(x, adj))) #20220122 #先activate再BatchNorm

        x = F.relu(self.gc5(x, adj)) #如果gc5是output layer
        #x = F.relu(self.apply_bn(self.gc5(x, adj))) #20220122 #先BatchNorm再activate

        #x = F.relu(self.gc6(x, adj)) #如果gc6是output layer
        
        #return F.log_softmax(x, dim=1) #original #cannot converge
        #return x #20220121 #when test, output identical values close to 0
        #return F.relu(x) #20220121 #when train, loss=nan, 调小lr可以缓解
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

    def apply_bn(self, x): #20220122 #BatchNorm
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.apply_bn(self.linear1(x))) #20220122 #先BatchNorm再activate
        x = F.relu(self.linear2(x))
        #x = F.relu(self.apply_bn(self.linear2(x))) #20220122 #先BatchNorm再activate
        x = self.linear3(x)
        return x


class PoolLayer(nn.Module): #20220120
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        x = ((x.T)*(x[:,:,-1].T)).T #mask
        x_avg =  (torch.sum(x[:,:,:-1],axis=1)/(len(torch.nonzero(x[0,:,-1],as_tuple=True)[0])))

        #x_avg =  torch.mean(x[torch.nonzero(x[:,:,-1],as_tuple=True)[0],torch.nonzero(x[:,:,-1],as_tuple=True)[1],:-1], axis=0).unsqueeze(axis=1)
        #x_std =  torch.std(x[torch.nonzero(x[:,:,-1],as_tuple=True)[0],torch.nonzero(x[:,:,-1],as_tuple=True)[1],:-1], axis=0).unsqueeze(axis=1)
        
        #x = x * (x[:,-1].unsqueeze(axis=1)) #mask
        #x_avg =  torch.mean(x[torch.nonzero(x[:,-1],as_tuple=True)[0],:-1], axis=0).unsqueeze(axis=1)
        #x_std =  torch.std(x[torch.nonzero(x[:,-1],as_tuple=True)[0],:-1], axis=0).unsqueeze(axis=1)
        #output = torch.cat((x_avg, x_std), dim=0).squeeze()
        output = x_avg
        return output



class GCN_OVER_MLP(nn.Module): #20220121
    def __init__(self, config):
        nn.Module.__init__(self)
        self.GCNLayer = GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout)
        self.PoolLayer = PoolLayer()
        self.MLPLayers = MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        self.dim_touched = config.dim_touched

    def forward(self, x, adj):
        #x = self.GCNLayer.forward(x, adj) 
        for i in range(x.shape[0]): #暂时无法批处理，只能土法循环  #20220121
            #this_output = self.GCNLayer.forward(x[i,:,:-1], adj) #最后一维标记是否免疫，不要动
            this_output = self.GCNLayer.forward(x[i,:,:self.dim_touched], adj) #20220123
            if(i==0):
                #all_gcn_output = this_output.clone()
                all_gcn_output = this_output.clone().unsqueeze(dim=0) #20220127
            #elif(i==1):
            #    all_gcn_output = torch.stack((all_gcn_output,this_output),dim=0)
            else:
                all_gcn_output = torch.cat((all_gcn_output,this_output.unsqueeze(dim=0)), dim=0)

        all_gcn_output = torch.cat((all_gcn_output, x[:,:,self.dim_touched:]), dim=2) #20220123

        x = self.PoolLayer.forward(all_gcn_output)
        x = self.MLPLayers.forward(x)
        return x


class Generator(nn.Module): #20220126
    def __init__(self, config):
        nn.Module.__init__(self)
        self.GCNLayer = GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout)
        #self.TopKPoolLayer = TopKPooling()
        self.MLPLayers = MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        self.dim_touched = config.dim_touched

    def forward(self, x, adj):
        all_gcn_output = self.GCNLayer.forward(x[:,:self.dim_touched], adj) 
        all_gcn_output = torch.cat((all_gcn_output, x[:,self.dim_touched:]), dim=1) #20220123
        x = self.MLPLayers.forward(all_gcn_output)
        pdb.set_trace()
        #x = x.squeeze()
        #predicted_sorted_indexes = torch.argsort(x)  # 返回从大到小的索引
        sorted_x = torch.topk(x.squeeze(),NN)
        return sorted_x #x


def get_model(config, model_name='GCN'):
    if(model_name=='GCN'):
        layers = Sequential(
            #GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout), #original
            GCN(config.gcn_nfeat, config.gcn_nhid1, config.gcn_nhid2, config.gcn_nclass, config.gcn_dropout), #20220119
            LinearLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    elif(model_name=='MLP'):
        layers = Sequential(
            PoolLayer(),
            MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
    elif(model_name=='GNN_OVER_MLP'): #20220121
        layers = GCN_OVER_MLP(config)
        '''
        layers = Sequential(
            GCN(config.gcn_nfeat, config.gcn_nhid, config.gcn_nclass, config.gcn_dropout),
            PoolLayer(),
            MLPLayers(config.linear_nin, config.linear_nhid1, config.linear_nhid2, config.linear_nout, config.linear_activation, config.linear_bias)
        )
        '''
    elif(model_name=='Generator'): #20220126
        layers = Generator(config)
    return layers