import torch
from torchvision import datasets, transforms

def train_loader(args, kwargs, dataset) :
    if dataset == 'mnist' :
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    else :
        print('There is no training dataset.')
        assert False
    
    return train_loader

def test_loader(args, kwargs, dataset) :
    if dataset == 'mnist' :
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    else :
        print('There is no training dataset.')
        assert False
    
    return test_loader