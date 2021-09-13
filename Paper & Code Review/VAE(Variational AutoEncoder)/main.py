from __future__ import print_function # python 2,3문법 호환용
import argparse
import torch
import numpy as np
import os

from torchvision.utils import save_image
from torch import optim
from VAE import VAE
from loss import loss_function
from train import train
from test import test
from dataloader import train_loader, test_loader
from utils import fix_randomness
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

fix_randomness(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = train_loader(args, kwargs, dataset='mnist')
test_loader = test_loader(args, kwargs, dataset='mnist')

os.environ['KMP_DUPLICATE_LIB_OK']='True' # plt error 방지

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, device, args, loss_function, epoch, train_loader)
        test(model, device, args, loss_function, epoch, test_loader)

        latent = []
        latent_mu = []
        targets = []
        i = 0
        with torch.no_grad():
            '''
            latent 시각화 :batch_size * 40 개의 samples를 뽑아서 확인
            '''
            vis_batch_num = 40
            for batch_x, batch_y in test_loader:
                mu, logvar = model.encode(batch_x.view(-1, 784))

                z = model.reparameterize(mu, logvar).detach().cpu().numpy()
                mu_samples = mu.detach().cpu().numpy()
                label = batch_y.detach().cpu().numpy()

                for x, m, y in zip(z, mu_samples, label) :
                    latent.append(x.reshape(2))
                    latent_mu.append(m.reshape(2))
                    targets.append(y)

                if i == vis_batch_num :
                    break
                
                i += 1

            '''
            Test data에 대한 latent space 2차원 시각화
            '''
            save_latent = './latent'
            os.makedirs(save_latent, exist_ok=True)
            latent = np.array(latent)
            latent_mu = np.array(latent_mu)
            plt.figure(figsize=(15,6))
            plt.subplot(1,2,1)
            plt.scatter(latent[:,0], latent[:,1], c=targets, cmap='jet')
            plt.title('Z Sample',fontsize=20);plt.colorbar();plt.grid()
            plt.xlim([-4,4]); plt.ylim([-4,4]);
            plt.subplot(1,2,2)
            plt.scatter(latent_mu[:,0], latent_mu[:,1], c=targets, cmap='jet')
            plt.title('Z mu',fontsize=20);plt.colorbar();plt.grid()
            plt.xlim([-4,4]); plt.ylim([-4,4]);
            plt.savefig(f'latent/{epoch}.jpg')

            '''
            Test data에 대한 Input vs Reconsturction 시각화
            '''
            n_sample = 5
            
            for batch_x, _ in test_loader :
                x_sample = batch_x
                recon_sample, _, _ = model(batch_x.view(-1, 784))
                break
            
            save_input = './input'
            os.makedirs(save_input, exist_ok=True)
            save_recon = './recon'
            os.makedirs(save_recon, exist_ok=True)

            fig = plt.figure(figsize=(15,3))
            for i in range(n_sample):
                plt.subplot(1,n_sample,i+1)
                plt.imshow(x_sample[i,:].reshape(28,28).detach().cpu().numpy(),vmin=0,vmax=1,cmap="gray")
            fig.suptitle("Training Inputs",fontsize=20);
            plt.savefig(f'input/{epoch}.jpg')
            
            fig = plt.figure(figsize=(15,3))
            for i in range(n_sample):
                plt.subplot(1,n_sample,i+1)
                plt.imshow(recon_sample[i,:].reshape(28,28).detach().cpu().numpy(),vmin=0,vmax=1,cmap="gray")
            fig.suptitle("Reconstructed Inputs",fontsize=20);
            plt.savefig(f'recon/{epoch}.jpg')
