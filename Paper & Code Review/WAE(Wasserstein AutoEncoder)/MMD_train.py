from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from kernel import imq_kernel, rbf_kernel
from prior import sample_pz

def MMD_train(encoder, decoder, enc_optim, dec_optim, args, train_loader, epoch) :
    encoder.train()
    decoder.train()

    REC_loss = 0
    MMD_loss = 0
    TOT_loss = 0
    
    for (images, _) in tqdm(train_loader) :
        images = images.cuda()
        enc_optim.zero_grad()
        dec_optim.zero_grad()

        z = encoder(images)
        x_recon = decoder(z)

        '''
        inf(E_pE_q|z(c(X, G(z)))) term
        '''
        criterion = nn.MSELoss() # cost function : L2-norm
        recon_loss = criterion(x_recon, images)

        '''
        Dz(Q, P) term : Maximum Mean Discrepancy(MMD)

        1) fake z : VAE와 달리 Gaussian 말고도 원하는 Prior을 사용가능함
        2) kernel trick을 사용하여 두 분포의 discrepancy를 계산함
        - 1. imq kernel
        - 2. gaussian kernel (rbf kernel)
        '''
        batch_size = images.size()[0]
        z_fake = Variable(torch.from_numpy(sample_pz(args.Pz, args.n_z, batch_size))).cuda()  # fake z 생성 : Batch, Latent_dim
        
        mmd_loss = imq_kernel(z, z_fake, h_dim=encoder.n_z, Pz=args.Pz) # input : Qz - encoder(X), Pz - fake z
        mmd_loss = args.Lambda * (mmd_loss / batch_size) # 전체 1/n 곱해주는 term

        total_loss = recon_loss + mmd_loss
        total_loss.backward()

        REC_loss += recon_loss.item()
        MMD_loss += mmd_loss.item()
        TOT_loss += total_loss

        enc_optim.step()
        dec_optim.step()

    print('====> Epoch: {} Average Recon loss: {:.4f}, Average MMD loss: {:.4f}, Average Total loss: {:.4f}'.format(
          epoch, REC_loss, MMD_loss, TOT_loss))