import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from MMD_train import MMD_train
from MMD_test import MMD_test
from GAN_train import GAN_train
from GAN_test import GAN_test
from dataloader import train_loader, test_loader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from utils import fix_randomness
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='WAE MNIST Example')
parser.add_argument('-batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=8, help='hidden dimension of z (default: 8)')
parser.add_argument('-Lambda', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
parser.add_argument('-Dz', type=str, default='MMD', help= 'MMD or GAN (default: MMD)')
parser.add_argument('-Pz', type=str, default='normal', help= 'normal or sphere (default: normal distribution)')
parser.add_argument('-no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('-seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

fix_randomness(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if args.Dz == 'MMD' :
    from MMD_model import Encoder, Decoder
else :
    from GAN_model import Encoder, Decoder, Discriminator

    discriminator = Discriminator(args).cuda()
    dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * args.lr)
    dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

encoder, decoder = Encoder(args), Decoder(args)

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)

# Scheduler
enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

if args.cuda :
    encoder, decoder = encoder.cuda(), decoder.cuda()

train_loader = train_loader(args, kwargs, dataset='mnist')
test_loader = test_loader(args, kwargs, dataset='mnist')

os.environ['KMP_DUPLICATE_LIB_OK']='True' # plt error 방지

if __name__ == '__main__' :
    best_loss = 1000000
    best_epochs = 1

    REC_loss_list = []
    Dz_loss_list = []
    TOT_loss_list = []

    for epoch in range(1, args.epochs + 1):
        latent = []
        targets = []

        i = 0
        if args.Dz == 'MMD' :
            MMD_train(encoder, decoder, enc_optim, dec_optim, args, train_loader, epoch)
            test_REC_loss, test_MMD_loss, test_TOT_loss = MMD_test(encoder, decoder, args, test_loader, epoch)

        elif args.Dz == 'GAN' :
            GAN_train(encoder, decoder, discriminator, enc_optim, dec_optim, dis_optim, args, train_loader, epoch)
            test_REC_loss, test_MMD_loss, test_TOT_loss = GAN_test(encoder, decoder, discriminator, args, test_loader, epoch)

        else :
            assert 'Dz is wrong'    
    
        REC_loss_list.append(test_REC_loss)
        Dz_loss_list.append(test_MMD_loss)
        TOT_loss_list.append(test_TOT_loss)

        if test_TOT_loss < best_loss :
            best_loss = test_TOT_loss
            best_epochs = epoch
            print('best_epochs_saved \n')

        with torch.no_grad():
            '''
            latent 시각화 : batch_size * 40 개의 samples를 뽑아서 확인
            '''
            vis_batch_num = 40
            for batch_x, batch_y in test_loader:
                latent_z = encoder(batch_x.cuda()).detach().cpu().numpy()
                label = batch_y.detach().cpu().numpy()

                for x, y in zip(latent_z, label) :
                    latent.append(x.reshape(2))
                    targets.append(y)

                if i == vis_batch_num :
                    break
                
                i += 1

            '''
            Test data에 대한 latent space 2차원 시각화
            '''
            save_latent = f'./latent/{args.Pz}_{args.Dz}'
            os.makedirs(save_latent, exist_ok=True)
            latent = np.array(latent)
            plt.figure(figsize=(15,13))
            # plt.subplot(1,2,1)
            plt.scatter(latent[:,0], latent[:,1], c=targets, cmap='jet')
            plt.title(f'Z Sample epochs : {epoch}',fontsize=20);plt.colorbar();plt.grid()
            plt.xlim([-4,4]); plt.ylim([-4,4]);
            plt.savefig(f'latent/{args.Pz}_{args.Dz}/{epoch}.jpg')

            '''
            Test data에 대한 Input vs Reconsturction 시각화
            '''
            n_sample = 5
            
            for batch_x, _ in test_loader :
                x_sample = batch_x
                z = encoder(batch_x.cuda())
                recon_sample = decoder(z)
                break
            
            save_input = f'./input/{args.Pz}_{args.Dz}'
            os.makedirs(save_input, exist_ok=True)
            save_recon = f'./recon/{args.Pz}_{args.Dz}'
            os.makedirs(save_recon, exist_ok=True)

            fig = plt.figure(figsize=(15,3))
            for i in range(n_sample):
                plt.subplot(1,n_sample,i+1)
                plt.imshow(x_sample[i,:].reshape(28,28).detach().cpu().numpy(),vmin=0,vmax=1,cmap="gray")
            fig.suptitle(f'Training Inputs epochs : {epoch}',fontsize=20);
            plt.savefig(f'input/{args.Pz}_{args.Dz}/{epoch}.jpg')
            
            fig = plt.figure(figsize=(15,3))
            for i in range(n_sample):
                plt.subplot(1,n_sample,i+1)
                plt.imshow(recon_sample[i,:].reshape(28,28).detach().cpu().numpy(),vmin=0,vmax=1,cmap="gray")
            fig.suptitle(f'Reconstructed Inputs epochs : {epoch}',fontsize=20);
            plt.savefig(f'recon/{args.Pz}_{args.Dz}/{epoch}.jpg')
    
    print(f'best_epochs : {best_epochs}, best_loss : {best_loss}')

    '''
    Loss Graph 시각화
    '''
    fig, loss_ax = plt.subplots()

    loss_ax.plot(REC_loss_list, 'b', label = 'REC loss')
    loss_ax.plot(Dz_loss_list, 'y', label = 'Dz loss')
    loss_ax.plot(TOT_loss_list, 'r', label = 'Total loss')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    loss_ax.legend(loc='upper left')
    # acc_ax.legend(loc='lower left')

    plt.savefig(f'loss_{args.Pz}_{args.Dz}.jpg')