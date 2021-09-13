import torch
from torchvision.utils import save_image

def test(model, device, args, loss_function, epoch, test_loader):
    model.eval()
    test_BCE_loss = 0
    test_KLD_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_BCE, test_KLD = loss_function(recon_batch, data, mu, logvar)
            test_BCE_loss += test_BCE.item()
            test_KLD_loss += test_KLD.item()

    test_data_len = len(test_loader.dataset)
    print('====> Test set BCE_loss: {:.4f}, KLD_loss: {:.4f}, Total_loss: {:.4f}'.format(
            test_BCE_loss / test_data_len, test_KLD_loss / test_data_len, (test_BCE_loss + test_KLD_loss) / test_data_len))