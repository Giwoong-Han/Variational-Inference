import torch
from torch.autograd import Variable
from kernel import imq_kernel, rbf_kernel
from prior import sample_pz
from torch.nn import functional as F
import torch.nn as nn


def MMD_test(encoder, decoder, args, test_loader, epoch) :
    encoder.eval()
    decoder.eval()

    test_REC_loss = 0
    test_MMD_loss = 0
    test_TOT_loss = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.cuda()
            z = encoder(images)
            x_recon = decoder(z)

            batch_size = images.size()[0]

            if args.dataset == 'mnist' : 
                recon_loss = F.binary_cross_entropy(x_recon, images, reduction='sum')
            else :
                criterion = nn.MSELoss() # cost function : L2-norm
                recon_loss = criterion(x_recon, images)

            z_fake = Variable(torch.from_numpy(sample_pz(args.Pz, args.n_z, batch_size))).cuda()

            mmd_loss = imq_kernel(z, z_fake, h_dim=encoder.n_z, Pz=args.Pz) # input : Qz - encoder(X), Pz - fake z
            mmd_loss = args.Lambda * (mmd_loss / batch_size)

            total_loss = recon_loss + mmd_loss
            
            test_REC_loss += recon_loss.item()
            test_MMD_loss += mmd_loss.item()
            test_TOT_loss += total_loss

    data_len = len(test_loader.dataset)

    print('====> Test set REC_loss: {:.4f}, MMD_loss: {:.4f}, Total_loss: {:.4f}'.format(
            test_REC_loss / data_len, test_MMD_loss / data_len, test_TOT_loss / data_len))

    return test_REC_loss / data_len, test_MMD_loss / data_len, test_TOT_loss / data_len
