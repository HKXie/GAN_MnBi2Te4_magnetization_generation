import numpy as np

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# functions to generate random data

def generate_random_image(img):
    random_data = torch.rand(*img.shape)#0到1之间随机取值
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
    
    def __init__(self, h_dim=500, in_channels=2, h_channels=64,learning_rate=1e-4, class_num=50, dropout_rate=0.1):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.convseq = nn.Sequential(
            
            nn.Conv2d(in_channels, h_channels, kernel_size=4, stride=2),#padding=1,#(_,_,20,20)->(_,_,9,9)
            # nn.AvgPool2d(in_channels, h_channels, kernel_size=3, stride=1),
            # nn.BatchNorm2d(h_channels),
            nn.LeakyReLU(),
            # nn.GELU(),
            
            nn.Conv2d(h_channels, h_channels, kernel_size=3, stride=1),#(_,_,9,9)->(_,_,7,7)
            # nn.AvgPool2d(in_channels, h_channels, kernel_size=3, stride=1),

            # nn.BatchNorm2d(h_channels),#6*6
            nn.LeakyReLU(),
            # nn.GELU(),
            
        )


        # 全连接层
        self.dense = nn.Sequential(

            nn.Linear(h_channels*7*7 + class_num, h_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(h_dim),

            nn.Dropout(p=dropout_rate),

            nn.Linear(h_dim, 2),
            # nn.Sigmoid()
        )


        
        # create loss function
        # self.loss_function = nn.BCELoss()
        self.loss_function = nn.CrossEntropyLoss()

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
    
    
    # def train(self, inputs, label_tensor, targets):
    #     # calculate the output of the network
    #     outputs = self.forward(inputs, label_tensor)
        
    #     # calculate loss
    #     loss = self.loss_function(outputs, targets)

    #     self.Loss=loss

    #     # zero gradients, perform a backward pass, update weights
    #     self.optimiser.zero_grad()
    #     loss.backward()
    #     self.optimiser.step()
def Train_D(model_D, optimiser, data_loader, loss_history, epoch, class_num):
    model_D.train()

    train_loss = 0
    Loss_D=0

    for image_data, label_tensor in data_loader:

        image_data = image_data.to(device)

        label_tensor = label_tensor.to(device)
        
        # real_imgs = torch.flatten(image_data, start_dim=1)#
        # real_imgs = image_data.view(-1,v_dim)

        # train real data
        outputs = model_D.forward(image_data, label_tensor)
        
        train_loss = model_D.loss_function(outputs, torch.ones(label_tensor.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([1.0])

        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        Loss_D += train_loss.detach().item()

        # train fake data

        outputs = model_D.forward(generate_random_image(image_data).cuda(), generate_random_one_hot(image_data.size(0), class_num).cuda())

        train_loss = model_D.loss_function(outputs, torch.zeros(label_tensor.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([0.0])

        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()
        Loss_D += train_loss.detach().item()
    
    if (epoch+1)%20==0 or epoch <= 5:
        print('Epoch {}:, Average train loss: {:.5f}'.format(epoch+1, Loss_D / (2*len(data_loader))))

    loss_history.append(Loss_D / (2*len(data_loader)))


def evaluate_D(model_D, data_loader, loss_history, Accuracy_rate, epoch, class_num):
    model_D.eval()

    total_samples = len(data_loader)*data_loader.batch_size#len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():

        for image_data, label_tensor in data_loader:

            image_data = image_data.to(device)

            label_tensor = label_tensor.to(device)
            
            # real_imgs = torch.flatten(image_data, start_dim=1)#
            # real_imgs = image_data.view(-1,v_dim)

            # train real data
            outputs = model_D.forward(image_data, label_tensor)
            
            train_loss = model_D.loss_function(outputs, torch.ones(label_tensor.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([1.0])


            total_loss += train_loss.detach().item()

            output_f=nn.Softmax(dim=-1)
            predicted = output_f(outputs).argmax(dim=-1)

            # total += target.size(0)
            correct_samples += predicted.eq(torch.ones(label_tensor.size(0)).cuda()).sum().item()


            # train fake data

            outputs = model_D.forward(generate_random_image(image_data).cuda(), generate_random_one_hot(label_tensor.size(0), class_num).cuda())

            train_loss = model_D.loss_function(outputs, torch.zeros(label_tensor.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([0.0])

            total_loss += train_loss.detach().item()

            output_f=nn.Softmax(dim=-1)
            predicted = output_f(outputs).argmax(dim=-1)

            # total += target.size(0)
            correct_samples += predicted.eq(torch.zeros(label_tensor.size(0)).cuda()).sum().item()

    avg_loss = total_loss / len(data_loader)#total_samples
    loss_history.append(avg_loss)
    Accuracy = (100.0 * correct_samples / (2*total_samples))
    Accuracy_rate.append(Accuracy)

 
    if (epoch+1) % 20 ==0 or epoch<=5:

        print('Epoch {}: '.format(epoch+1)+'Average test loss: ' + '{:.5f}'.format(avg_loss) +
            '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
            '{:5}'.format(total_samples*2) + ' (' +
            '{:4.2f}'.format(Accuracy) + '%)\n')





# generator class

class Generator(nn.Module):
    
    def __init__(self, h_dim=500, z_dim=100, out_channels=2, h_channels=64,learning_rate=1e-4, class_num=50, dropout_rate=0.1):
        # initialise parent pytorch class
        super().__init__()

        self.out_channels=out_channels
        self.h_channels=h_channels
        
        # define neural network layers
        self.dense = nn.Sequential(


            nn.Linear(z_dim+class_num, h_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(h_dim),

            nn.Dropout(p=dropout_rate),  # 缓解过拟合，一定程度上正则化原文链接：https://blog.csdn.net/m0_62001119/article/details/121757703

            # nn.Linear(h_dim, h_dim*2),
            # nn.LeakyReLU(),
            # # nn.LayerNorm(h_dim),
            # nn.Dropout(p=dropout_rate), 

            nn.Linear(h_dim, h_channels*7*7),
            nn.LeakyReLU(),
            nn.LayerNorm(h_channels*7*7),


           
        )

        self.convtra = nn.Sequential(
           
            nn.ConvTranspose2d(h_channels, h_channels, kernel_size=3, stride=1),#2*10*10

            # nn.BatchNorm2d(h_channels),
            nn.LeakyReLU(),
            # nn.GELU(),

            nn.ConvTranspose2d(h_channels, out_channels, kernel_size=4, stride=2),
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
        outputs = outputs.view(-1, self.h_channels, 7, 7)
        return self.convtra(outputs)
    
    
    # def train(self, D, inputs, label_tensor, targets):
    #     # calculate the output of the network
    #     g_output = self.forward(inputs, label_tensor)
        
    #     # pass onto Discriminator
    #     d_output = D.forward(g_output, label_tensor)
        
    #     # calculate error
    #     loss = D.loss_function(d_output, targets)

    #     self.Loss=loss#用于记录loss

    #     # zero gradients, perform a backward pass, update weights
    #     self.optimiser.zero_grad()
    #     loss.backward()
    #     self.optimiser.step()

def Train_G(z_dim, G_model, D_model, optimiser_D, optimiser_G, data_loader, D_loss_history, G_loss_history, epoch, class_num):

    D_model.train()
    G_model.train()

    d_losses=0
    g_losses=0


    for image_data, label_tensor in data_loader:#data_set
    
        # image_data_tensor = image_data_tensor.view(-1, v_dim)#torch.flatten(image_data_tensor)#, start_dim=1

        image_data = image_data.to(device)

        label_tensor = label_tensor.to(device)
        

# train real data
        outputs = D_model.forward(image_data, label_tensor)
        
        train_loss = D_model.loss_function(outputs, torch.ones(image_data.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([1.0])

        optimiser_D.zero_grad()
        train_loss.backward()
        optimiser_D.step()

        d_losses += train_loss.detach().item()


# train discriminator by Generator

        Genetated_random_hot=generate_random_one_hot(image_data.size(0), class_num)

        g_image = G_model.forward(generate_random_seed(image_data.size(0), z_dim).cuda(), Genetated_random_hot.cuda())

        g_outputs = D_model.forward(g_image.detach(), Genetated_random_hot.cuda())
        # use detach() so gradients in G are not calculated

        train_loss = D_model.loss_function(g_outputs, torch.zeros(image_data.size(0)).to(torch.long).cuda())#torch.cuda.FloatTensor([1.0])

        optimiser_D.zero_grad()
        train_loss.backward()
        optimiser_D.step()

        d_losses += train_loss.detach().item()


# # train generator

        Genetated_random_hot=generate_random_one_hot(image_data.size(0), class_num)

        g_image = G_model.forward(generate_random_seed(image_data.size(0), z_dim).cuda(), Genetated_random_hot.cuda())
        
        # pass onto Discriminator
        DG_outputs = D_model.forward(g_image, Genetated_random_hot.cuda())
        
        # calculate error
        train_loss = D_model.loss_function(DG_outputs, torch.ones(image_data.size(0)).to(torch.long).cuda())

        optimiser_G.zero_grad()
        train_loss.backward()
        optimiser_G.step()

        g_losses += train_loss.detach().item()

    
    if (epoch+1) % 20 ==0 or epoch<=5:
        print('Epoch: {}, D_loss: {}, G_loss: {}'.format(epoch+1, d_losses / (2*len(data_loader)),g_losses / len(data_loader)))

    D_loss_history.append(d_losses / (2*len(data_loader)))
    G_loss_history.append(g_losses / len(data_loader))
