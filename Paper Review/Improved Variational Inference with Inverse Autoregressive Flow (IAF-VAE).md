# Improved Variational Inference with Inverse Autoregressive Flow

- Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling
- Accepted at 30th Conference on Neural Information Processing Systems (NIPS 2016)

## Short Review

1. 해당 방법론은 VAE의 저자가 보다 Posterior에 대해 Variational inference 적용 시 Flexible한 Latent를 만들도록 제안한 연구입니다.

2. VAE의 다양한 갈래 중 Normalizing Flow에 관련된 연구입니다.

## Methods

1. Normalizing Flow (NF)
- ELBO의 문제점에 대한 그림
- 수식 설명

2. Inverse Autoregressive Flow (IAF)
- 구조
- LSTM idea 설명
- 최종 loss
