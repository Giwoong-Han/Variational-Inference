import torch
from torch.nn import functional as F
from torch import nn, optim

class VAE(nn.Module):
    '''
    Architecture
    - MNIST에서 주로 사용하는 Basic NN
    - 784(input) -> ReLU(400)-> 2(mu), 2(sigma) -> z = mu + sigma * epsilon (2 dim) -> ReLU(400) -> 784(recon)
    - linear layer를 통과하기 때문에, Activation fuction을 ReLU를 사용
    '''
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # std = sigma
        eps = torch.randn_like(std) # std와 같은 사이즈 표준정규분포를 따르는 random값 생성
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784)) # batch, dim으로 인풋으로 들어감 (Reshape)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar