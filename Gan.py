import torch
import torch.nn as nn
import numpy as np

# functions to generate random data

def generate_random_image(batch_size,size):
    random_data = torch.rand(batch_size,size)#0到1之间随机取值
    return random_data


def generate_random_seed(batch_size,size):
    random_data = torch.randn(batch_size,size)#标准正态分布
    return random_data

def generate_random_one_hot(batch_size,size):
    label_tensor=torch.zeros(batch_size,size)
    for i in range(batch_size):
        randon_idx = np.random.randint(0,size)
        label_tensor[i, randon_idx]=1.0
    return label_tensor


# discriminator class

class Discriminator(nn.Module):
    
    def __init__(self, v_dim=2*10*10, h_dim=500, z_dim=100, learning_rate=1e-4, class_num=50):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            
            nn.Linear(v_dim+class_num, h_dim),
            nn.LeakyReLU(),           
            nn.LayerNorm(h_dim),
            
            nn.Linear(h_dim, 1000),
            nn.LeakyReLU(),
            nn.LayerNorm(1000),


            # nn.Linear(h_dim, z_dim),
            # nn.LeakyReLU(),
            
            # nn.LayerNorm(z_dim),
            
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )
        
        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # counter and accumulator for progress
        # self.counter = 0;
        # self.progress = []

        self.Loss=0

        pass
    
    
    def forward(self, inputs, label_tensor):
        inputs=torch.cat((inputs,label_tensor),axis=1)
        # simply run model
        return self.model(inputs)
    
    
    def train(self, inputs, label_tensor ,targets):
        # calculate the output of the network
        outputs = self.forward(inputs,label_tensor)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        self.Loss=loss

        # increase counter and accumulate error every 10
        # self.counter += 1;

        # if (self.counter % 10 == 0):
        
        #     self.progress.append(loss.item())
        
        # if (self.counter % 500 == 0):
        #     print("counter = ", self.counter)
        # #     pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()



# generator class

class Generator(nn.Module):
    
    def __init__(self,v_dim=2*10*10, h_dim=500, z_dim=100, learning_rate=1e-4, class_num=50):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(

            nn.Linear(z_dim+class_num, 1000+500),
            nn.LeakyReLU(),           
            nn.LayerNorm(1000+500),

            nn.Linear(1000+500, h_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(h_dim),

            
            nn.Linear(h_dim, v_dim),
            
            nn.Sigmoid(),
        )
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # counter and accumulator for progress

        self.Loss=0
        
    
    
    def forward(self, inputs, seed_tensor):        
        # simply run model
        inputs=torch.cat((inputs, seed_tensor),axis=1)
        return self.model(inputs)
    
    
    def train(self, D, inputs, label_tensor, targets):
        # calculate the output of the network
        g_output = self.forward(inputs, label_tensor)
        
        # pass onto Discriminator
        d_output = D.forward(g_output, label_tensor)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        self.Loss=loss#record loss

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()