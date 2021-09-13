import torch
from torch.nn import functional as F

def loss_function(recon_x, x, mu, logvar):
    '''
    방법에 따라 lambda * BCE - (1-lambda) * KLD로 loss fuction에 가중치를 줄 수도 있음

    1. BCE : L(y_i, t_i) = - 1/N * sum_i=1 to N (t_i * log(y_i) + (1 - t_i) * log(1 - y_i))
    - N : Batch Size
    - reduction : mean, sum -> 최종 output loss 값을 Batch의 평균 or 합 옵션
    - 예측값과 실제값이 같은 경우 : 0, 다른 경우 (차이가 클수록 무한대로 발산)

    2. KLD : D_KL(P(z)||Q(z)) = integral(Q(z) * log(P(z)/Q(z)) dz)
    - Prior dist ~ z 가 mu 0이고 cov = I 인 Isotropic한 분포를 따르므로 다음과 같이 analytic한 form으로 계산 가능
    '''
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') # L2-norm : |x - x^hat|_2 대신 사용가능 (image : 0 ~ 1 normalize 한 경우만)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD