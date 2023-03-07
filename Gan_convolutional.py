import numpy as np

import torch
import torch.nn as nn

# functions to generate random data

def generate_random_image(batch_size):
    random_data = torch.rand(batch_size,2,10,10)#0到1之间随机取值
    return random_data


def generate_random_seed(batch_size, z_dim):
    random_data = torch.randn(batch_size, z_dim)#标准正态分布
    return random_data


def generate_random_one_hot(batch_size,size):
    label_tensor=torch.zeros(batch_size,size)
    for i in range(batch_size):
        randon_idx=np.random.randint(0,size)
        label_tensor[i, randon_idx]=1.0
    return label_tensor

# discriminator class

class Discriminator(nn.Module):
    
    def __init__(self,h_dim=500, learning_rate=1e-4, class_num=50):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.convseq = nn.Sequential(
            
            nn.Conv2d(2, 64, kernel_size=3, stride=1),#padding=1,这里尽量设置不补
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),#8*8
            # nn.GELU(),
            
            nn.Conv2d(64, 24, kernel_size=3, stride=1),
            # nn.BatchNorm2d(24),#6*6
            nn.LeakyReLU(),
            # nn.GELU(),
            
        )


        # 全连接层
        self.dense = nn.Sequential(

            nn.Linear(24*6*6 + class_num, h_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(h_dim),

            # nn.Dropout(p=0.5),  # 缓解过拟合，一定程度上正则化原文链接：https://blog.csdn.net/m0_62001119/article/details/121757703

            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )


        
        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # counter and accumulator for progress

        self.Loss=0

        pass
    
    
    def forward(self, inputs, label_tensor):
        # simply run model
        outputs=self.convseq(inputs)
        outputs = outputs.view(outputs.size(0), -1) 
        outputs=torch.cat((outputs,label_tensor),axis=1)

        return self.dense(outputs)
    
    
    def train(self, inputs, label_tensor, targets):
        # calculate the output of the network
        outputs = self.forward(inputs, label_tensor)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        self.Loss=loss

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


# generator class

class Generator(nn.Module):
    
    def __init__(self,h_dim=200, z_dim=100, learning_rate=1e-4, class_num=50):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.dense = nn.Sequential(
            # 线性分类器
            # []
            nn.Linear(z_dim+class_num, h_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(h_dim),

            # nn.Dropout(p=0.5),  # 缓解过拟合，一定程度上正则化原文链接：https://blog.csdn.net/m0_62001119/article/details/121757703

            # nn.Linear(h_dim, h_dim),
            # nn.LeakyReLU(),
            # nn.LayerNorm(h_dim),

            nn.Linear(h_dim, 24*6*6),
            nn.LeakyReLU(),
            nn.LayerNorm(24*6*6),


           
        )

        self.convtra = nn.Sequential(
           
            nn.ConvTranspose2d(24, 64, kernel_size=3, stride=1),#2*10*10

            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # nn.GELU(),

            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1),
            # nn.BatchNorm2d(2),
            
            # output should be (1,2,10,10)
            nn.Sigmoid()
        )
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.Loss=0
        
    
    
    def forward(self, inputs, seed_tensor):        
        # simply run model
        inputs=torch.cat((inputs, seed_tensor),axis=1)
        outputs = self.dense(inputs)
        outputs = outputs.view(-1,24,6,6)
        return self.convtra(outputs)
    
    
    def train(self, D, inputs, label_tensor, targets):
        # calculate the output of the network
        g_output = self.forward(inputs, label_tensor)
        
        # pass onto Discriminator
        d_output = D.forward(g_output, label_tensor)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        self.Loss=loss#用于记录loss

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
