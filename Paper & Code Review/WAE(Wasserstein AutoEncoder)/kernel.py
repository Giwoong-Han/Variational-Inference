import torch

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int,
               Pz):
    '''
    Inverse MultiQuadric (IMQ) :

    k(x, y) = sum(s*C/{s*C + ||x - y||^2}) where s in {0.1, 0.2, 0.5, 1, 2, 5, 10}, C was sampled from U[0.1, 16]
    MMD에서 사용 -> Kernel Trick : MMD(P,Q) : E(k(X,X')) + E(k(Y,Y')) - 2E(k(X,Y))
    '''
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    if Pz == 'normal' :
        Cbase = 2. * h_dim  # sigma = 1
    else :
        Cbase = 2.
    stats = 0        
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    '''
    Gaussian Kernel :
    MMD에서 사용 -> Kernel Trick : MMD(P,Q) : E(k(X,X')) + E(k(Y,Y')) - 2E(k(X,Y))
    '''
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats