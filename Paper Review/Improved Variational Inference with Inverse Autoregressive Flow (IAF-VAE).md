# Improved Variational Inference with Inverse Autoregressive Flow

- Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling
- Accepted at 30th Conference on Neural Information Processing Systems (NIPS 2016)

## Short Review

1. 해당 방법론은 VAE의 저자가 보다 Posterior에 대해 Variational inference 적용 시 Flexible한 Latent를 만들도록 제안한 연구입니다.

2. VAE의 다양한 갈래 중 Normalizing Flow에 관련된 연구입니다.

## Methods

1. Normalizing Flow (NF)

> * 우선 14년 저자가 발표했던 VAE의 ELBO (Evidence Lower Bounded)와 LLE(Log Likeliwood)를 시각화한 그림입니다.

<br>
<그림 추가>
<br>

> * Normalizing Flow는 위 그림에서 보여지는 부분과 같이 Inference 부분에서 Flexible한 Latent Space를 잘 만들어야 LLE와 ELBO의 차이를 줄일 수 있다는 생각해서 시작합니다. 따라서 기존에 학습한 Latent Space에 Invertible한 Transformation Function을 T번을 반복하여 Flexible하게 변형된 Latent를 사용하겠다는 것입니다.

<br>

<수식 추가 4, 5>
> * 위와 같이 Jacobian Determinant를 각 반복 횟수마다 구할 수 있으므로 최종 시점 T에서의 학습이 가능하도록 설계할 수 있습니다. 그리고 역변환이 가능한 Jacobian Determinant라는 가정이 포함되면 아래와 같이 아주 단순한 수식으로 Transformation Function을 만들 수 있습니다.
<br>
<수식 추가 6>
<br>

참고 : https://www.ritchievink.com/blog/2019/11/12/another-normalizing-flow-inverse-autoregressive-flows/
https://bjlkeng.github.io/posts/variational-autoencoders-with-inverse-autoregressive-flows/
2. Inverse Autoregressive Transformation
> * NF -> Inverse 적용한 부분 설명

3. Inverse Autoregressive Flow (IAF)
> * 구조
> * LSTM idea 설명
> * 최종 loss
