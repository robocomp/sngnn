import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from functools import partial

class RGCNLayer(nn.Module):
    def __init__(self, g, in_feat, out_feat, num_rels, feat_drop, num_bases=-1, activation=None, freeze=(0,0)):
        super(RGCNLayer, self).__init__()
        self.g = g
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = activation
        self.freeze = freeze
        #print('BASES', self.num_bases)
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        # print("in_feat:{}, out_feat:{}, num_rels:{}".format(self.in_feat, self.out_feat, self.num_rels, end=' '))
        # print("num_bases:{}, activation:{}".format(self.num_bases, self.activation))

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat+self.freeze[0], self.out_feat))

        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('elu'))
        nn.init.xavier_normal_(self.weight, gain=1.414)

        if self.num_bases < self.num_rels:
            # nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('elu'))
            nn.init.xavier_normal_(self.w_comp, gain=1.414)

    def forward(self, inputs):
        inputs = self.feat_drop(inputs)
        self.g.ndata.update({'h': inputs})
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.in_feat+self.freeze[0], self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat+self.freeze[0], self.out_feat)
        else:
            weight = self.weight

        def message_func(edges):
            w = weight[edges.data['rel_type'].squeeze()]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1)
            edges.data['norm'] = edges.data['norm'].to(device=msg.device)
            msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.activation:
                h = self.activation(h)
            if self.freeze[1] > 0:
                return {'h': torch.cat((self.g.ndata['frozen'], h), 1) }
            return {'h': h }

        self.g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
        return self.g.ndata['h']


class RGCN(nn.Module):
    def __init__(self, g, in_dim, h_dim, out_dim, num_rels, feat_drop, num_bases=-1, num_hidden_layers=1, freeze=0):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.freeze = freeze
        # create RGCN layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        #print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.h_dim))
        return RGCNLayer(self.g, self.in_dim, self.h_dim, self.num_rels, self.feat_drop, self.num_bases,  activation=F.leaky_relu, freeze=(0,self.freeze)) # elu?

    def build_hidden_layer(self):
        #print('Building an HIDDEN  layer of {}x{}'.format(self.h_dim, self.h_dim))
        return RGCNLayer(self.g, self.h_dim, self.h_dim,  self.num_rels, self.feat_drop, self.num_bases,  activation=F.leaky_relu, freeze=(self.freeze,self.freeze)) # elu?

    def build_output_layer(self):
        #print('Building an OUTPUT  layer of {}x{}'.format(self.h_dim, self.out_dim))
        return RGCNLayer(self.g, self.h_dim, self.out_dim, self.num_rels, self.feat_drop, self.num_bases, activation=torch.tanh, freeze=(self.freeze,0))

    def forward(self, features):
        h = features
        self.g.ndata['frozen'] = h[:, :self.freeze]
        self.g.ndata['h'] = features
        for layer in self.layers:
            h = layer(h)
        ret = self.g.ndata.pop('h')
        # print ('RET: ({})'.format(ret.shape))
        return ret
