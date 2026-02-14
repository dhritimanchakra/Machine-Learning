import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

sns.set_theme()

(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255
testX = np.float32(testX) / 255

class Adam:
    def __init__(self,model,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.model=model
        self.mt=[torch.zeros_like(p) for p in model.parameters()]
        self.vt=[torch.zeros_like(p) for p in model.parameters()]
        self.t=0
        self.beta1=beta1
        self.beta2=beta2
        self.alpha=alpha
        self.epsilon=epsilon
    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad=torch.zeros_like(p.grad)
    def step(self):
        self.t += 1
        b1 = self.beta1
        b2 = self.beta2
        lr = self.alpha
        eps = self.epsilon
        t = self.t

        for i, p in enumerate(self.model.parameters()):
            if p.grad is None:
                continue
            g = p.grad.data
            self.mt[i] = b1 * self.mt[i] + (1 - b1) * g
            self.vt[i] = b2 * self.vt[i] + (1 - b2) * (g * g)

            m_hat = self.mt[i] / (1 - b1 ** t)
            v_hat = self.vt[i] / (1 - b2 ** t)

            with torch.no_grad():
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps)

def train(model,optimizer):
    testing_accuracy=[]
    for epoch in range(nb_epochs):
        indices = torch.randperm(trainX.shape[0])[:batch_size]
        x=trainX[indices].reshape(-1,28*28)
        y=trainy[indices]
        log_prob=model(torch.from_numpy(x).to(device))
        loss=loss_fct(log_prob,torch.from_numpy(y).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%100==0:
            model.train(mode=False)
            log_prob=model(torch.from_numpy(testX.reshape(-1,28*28))).to(device)
            testing_accuracy.append(
                (log_prob.argmax(-1) == torch.from_numpy(testy).to(device)).sum().item() / testy.shape[0])
            model.train(mode=True)

    return testing_accuracy
    
if __name__=="__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    labels=["Pytorch adam","THis implementation"]
    for i,optim in enumerate([torch.optim.Adam,Adam]):
        model=torch.nn.Sequential(nn.Dropout(p=0.4),
                                  nn.Linear(28*28,1200),
                                  nn.Dropout(p=0.4),
                                  nn.Linear(1200,10),
                                  nn.LogSoftmax(dim=-1)).to(device)
        optimizer=optim(model if i==1 else model.parameters())
        testing_accuracy=train(model,optimizer, nb_epochs=1000)
        plt.plot(testing_accuracy,label=labels[i])
    plt.legend()
    plt.xlabel('Epochs (x100)')
    plt.ylabel('Testing accuracy', fontsize=14)
    plt.savefig('adam.png', bbox_inches='tight')
    plt.show()