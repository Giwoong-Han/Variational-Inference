import torch
from torch.autograd import Variable
from kernel import imq_kernel, rbf_kernel
from prior import sample_pz
from torch.nn import functional as F
import torch.nn as nn

def GAN_test(encoder, decoder, discriminator, args, test_loader, epoch) :
    encoder.eval()
    decoder.eval()

    test_REC_loss = 0
    test_Dz_loss = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            batch_size = images.size()[0]
            images = images.cuda()
            z = encoder(images)
            x_recon = decoder(z)

            if args.dataset == 'mnist' : 
                recon_loss = F.binary_cross_entropy(x_recon, images, reduction='sum')
            else :
                criterion = nn.MSELoss() # cost function : L2-norm
                recon_loss = criterion(x_recon, images)

            d_real = discriminator(encoder(Variable(images.data)))
            z_fake = Variable(torch.from_numpy(sample_pz(args.Pz, args.n_z, batch_size))).cuda()  # fake z 생성 : Batch, Latent_dim
            d_fake = discriminator(z_fake)
            dis_loss = (torch.log(d_fake) + torch.log(1 - d_real) * args.Lambda).mean()
            
            test_REC_loss += recon_loss.item()
            test_Dz_loss += -dis_loss.item()

    data_len = len(test_loader.dataset)

    print('====> Test set REC_loss: {:.4f}, Dz_loss: {:.4f}, Total_loss: {:.4f}'.format(
            test_REC_loss / data_len, test_Dz_loss / data_len, (test_REC_loss + test_Dz_loss) / data_len))

    return test_REC_loss / data_len, test_Dz_loss / data_len, (test_REC_loss + test_Dz_loss) / data_len