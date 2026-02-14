import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data  

(trainX, trainy), (testX, testy) = load_data()
trainX=(np.float32(trainX)-127.5)/127.5


def get_minibatch(batch_size):
    indices=torch.randperm(trainX.shape[0])[:batch_size]
    return torch.tensor(trainX[indices],dtype=torch.float).reshape(batch_size,-1)

def sample_noise(size,dim=100):
    out=torch.empty(size,dim)
    mean=torch.zeros(size,dim)
    std=torch.ones(dim)
    torch.normal(mean,std,out=out)
    return out

class Generator(nn.Module):
    def __init__(self,input_dim=100,hidden_dim=1200,output_dim=28*28):
        super(Generator,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2,output_dim),
            nn.Tanh(),

        )
    def forward(self,noise):
        return self.network(noise)

class Discriminator(nn.Module):
    def __init__(self,input_dim=28*28,hidden_dim=1200,output_dim=1):
        super(Discriminator,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2,output_dim),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.network(x)
    
def train(generator,discriminator,generator_optimizer,discriminator_optimizer,nb_epoch,k=1,batch_size=100):
    criterion=torch.nn.BCELoss()
    training_loss={'generative':[],"discriminative":[]}
    for i in tqdm(range(nb_epoch)):
        for j in range(k):
            z=sample_noise(batch_size)
            x=get_minibatch(batch_size)
            real_preds=discriminator(x).squeeze()
            fake_preds=discriminator(generator(z).detach()).squeeze()
            r_loss=criterion(real_preds,torch.ones_like(real_preds))
            f_loss=criterion(fake_preds,torch.zeros_like(fake_preds))
            d_loss=(r_loss+f_loss)/2
            discriminator_optimizer.zero_grad()
            d_loss.backward()
            discriminator_optimizer.step()
            training_loss['discriminative'].append(d_loss.item())
        z=sample_noise(batch_size)
        fake_preds=discriminator(generator(z)).view(-1)
        g_loss=criterion(fake_preds,torch.ones(batch_size))
        generator_optimizer.zero_grad()
        g_loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(g_loss.item())
    return training_loss
        
if  __name__=="__main__":
    discriminator=Discriminator()
    generator=Generator()
    optimizer_d=optim.Adam(discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
    optimizer_g=optim.Adam(generator.parameters(),lr=0.0002,betas=(0.5,0.999))
    loss=train(generator,discriminator,optimizer_g,optimizer_d,5000,batch_size=100)
    NB_IMAGES=25
    z=sample_noise(NB_IMAGES)
    x=generator(z)
    plt.figure(figsize=(17,17))
    for i in range(NB_IMAGES):
        plt.subplot(5,5,1+i)
        plt.axis('off')
        plt.imshow(x[i].data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.savefig("Imgs/regenerated_MNIST_data.png")
    plt.show()


