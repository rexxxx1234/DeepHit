'''
This declare DeepHit architecture:

INPUTS:
    - input_dims: dictionary of dimension information
        > x_dim: dimension of features
        > num_Event: number of competing events (this does not include censoring label)
        > num_Category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
                      : this is equivalent to the output dimension
    - network_settings:
        > h_dim_shared & num_layers_shared: number of nodes and number of fully-connected layers for the shared subnetwork
        > h_dim_CS & num_layers_CS: number of nodes and number of fully-connected layers for the cause-specific subnetworks
        > active_fn: 'relu', 'elu', 'tanh'
        > initial_W: Xavier initialization is used as a baseline

LOSS FUNCTIONS:
    - 1. loglikelihood (this includes log-likelihood of subjects who are censored)
    - 2. rankding loss (this is calculated only for acceptable pairs; see the paper for the definition)
    - 3. calibration loss (this is to reduce the calibration loss; this is not included in the paper version)
'''

import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

class Model_DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(Model_DeepHit, self).__init__()

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']

        # NETWORK HYPER-PARMETERS
        self.h_dim_shared       = network_settings['h_dim_shared']
        self.h_dim_CS           = network_settings['h_dim_CS']
        self.num_layers_shared  = network_settings['num_layers_shared']
        self.num_layers_CS      = network_settings['num_layers_CS']

        self.active_fn          = network_settings['active_fn']
        self.drop_rate          = network_settings['drop_rate']

        layers = []

        #shared FCN
        for layer in range(self.num_layers_shared):
            if self.num_layers_shared == 1:
                layers.extend([nn.Linear(self.x_dim, self.h_dim_shared), self.active_fn])
            else:
                if layer == 0:
                    layers.extend([nn.Linear(self.x_dim, self.h_dim_shared),
                                   self.active_fn, nn.Dropout(p=self.drop_rate)])
                if layer >= 0 and layer != (self.num_layers_shared - 1):
                    layers.extend([nn.Linear(self.h_dim_shared, self.h_dim_shared),
                                   self.active_fn, nn.Dropout(p=self.drop_rate)])
                else:  # layer == num_layers-1 (the last layer)
                    layers.extend([nn.Linear(self.h_dim_shared, self.h_dim_shared), self.active_fn])

        self.shared = nn.Sequential(*layers)
        self.shared.apply(init_weights)


        layers = []

        #CS FCN
        for layer in range(self.num_layers_CS):
            if self.num_layers_CS == 1:
                layers.extend([nn.Linear(self.x_dim+self.h_dim_shared, self.h_dim_CS), self.active_fn])
            else:
                if layer == 0:
                    layers.extend([nn.Linear(self.x_dim+self.h_dim_shared,
                                   self.h_dim_CS), self.active_fn, nn.Dropout(p=self.drop_rate)])
                if layer >= 0 and layer != (self.num_layers_CS - 1):
                    layers.extend([nn.Linear(self.h_dim_CS, self.h_dim_CS),
                                   self.active_fn, nn.Dropout(p=self.drop_rate)])
                else:  # layer == num_layers-1 (the last layer)
                    layers.extend([nn.Linear(self.h_dim_CS, self.h_dim_CS), self.active_fn])

        self.CS = nn.Sequential(*layers)
        self.CS.apply(init_weights)

        layers = []

        #Outputs Layer
        layers.extend([nn.Dropout(p=self.drop_rate), nn.Linear(self.num_Event*self.h_dim_CS,
                        self.num_Event*self.num_Category), nn.Softmax(dim=-1)])

        self.out = nn.Sequential(*layers)
        self.out.apply(init_weights)

    def forward(self, x):
        shared_out = self.shared(x)
        h = torch.cat([x, shared_out], dim=1)

        out = []
        for _ in range(self.num_Event):
            cs_out = self.CS(h)
            out.append(cs_out)

        out = torch.stack(out, dim=1) # stack referenced on subject
        out = torch.reshape(out, [-1, self.num_Event*self.h_dim_CS])
        out = self.out(out)
        out = torch.reshape(out, [-1, self.num_Event, self.num_Category])

        return out


