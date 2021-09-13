def train(model, optimizer, device, args, loss_function, epoch, train_loader):
    model.train()
    BCE_loss = 0
    KLD_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE + KLD 
        loss.backward()
        BCE_loss += BCE.item()
        KLD_loss += KLD.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            batch_len = len(data)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t BCE_Loss: {:.6f}, KLD_Loss: {:.6f}, Total_Loss: {:.6f}'.format(
                epoch, batch_idx * batch_len, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), BCE.item() / batch_len, KLD.item() / batch_len, loss.item() / batch_len))
    
    data_len = len(train_loader.dataset)
    print('====> Epoch: {} Average BCE loss: {:.4f}, Average KLD loss: {:.4f}, Average Total loss: {:.4f}'.format(
          epoch, BCE_loss / data_len, KLD_loss / data_len, (BCE_loss + KLD_loss) / data_len))