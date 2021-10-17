from torchvision.transforms.transforms import Lambda
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from prior import sample_pz
from torch.nn import functional as F

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def GAN_train(encoder, decoder, discriminator, enc_optim, dec_optim, dis_optim, args, train_loader, epoch) :
    encoder.train()
    decoder.train()
    discriminator.train()

    REC_loss = 0
    Dz_loss = 0

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()
    
    for (images, _) in tqdm(train_loader) :
        images = images.cuda()
        enc_optim.zero_grad()
        dec_optim.zero_grad()
        discriminator.zero_grad()

        batch_size = images.size()[0]
        '''
        Discriminator train
        - encoder, decoder 막고, discriminator만 학습
        '''
        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        z_fake = Variable(torch.from_numpy(sample_pz(args.Pz, args.n_z, batch_size))).cuda()  # fake z 생성 : Batch, Latent_dim
        d_fake = discriminator(z_fake)

        z_real = encoder(images)
        d_real = discriminator(z_real)

        '''
        mean(log(Dr(z_fake)) + log(1-Dr(z_real)))
        '''

        dis_loss = (torch.log(d_fake) + torch.log(1 - d_real) * args.Lambda).mean()
        dis_loss.backward(mone)

        dis_optim.step()

        '''
        Generator train
        - discriminator 막고, encoder, decoder만 학습
        '''
        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        z_real = encoder(images)
        x_recon = decoder(z_real)
        d_real = discriminator(encoder(Variable(images.data)))

        '''
        mean(c(x, G(z_real)) - lambda * log(Dr(z_real)))
        '''
        if args.dataset == 'mnist' : 
            recon_loss = F.binary_cross_entropy(x_recon, images, reduction='sum')
        else :
            criterion = nn.MSELoss() # cost function : L2-norm
            recon_loss = criterion(x_recon, images)
        d_loss = args.Lambda * (torch.log(d_real)).mean()

        recon_loss.backward(one)
        d_loss.backward(mone)

        enc_optim.step()
        dec_optim.step()

        REC_loss += recon_loss.item()
        Dz_loss += -dis_loss.item()

        enc_optim.step()
        dec_optim.step()

    data_len = len(train_loader.dataset)

    print('====> Epoch: {} Average Recon loss: {:.4f}, Average Dz loss: {:.4f}, Average Total loss: {:.4f}'.format(
          epoch, REC_loss / data_len, Dz_loss / data_len, (REC_loss+Dz_loss) / data_len))