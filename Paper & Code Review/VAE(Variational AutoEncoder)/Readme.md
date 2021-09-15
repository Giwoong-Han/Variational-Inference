# Auto-Encoding Variational Bayes

- Diederik P Kingma, Max Welling

## Short Review

1. 본 논문은 Variational Inference, Bayesian Deep Learning을 공부할 때 가장 기본적으로 알아야 할 개념들을 소개합니다.

<br>

2. The Variational bound : VAE를 설계하는 과정에서 가장 중요한 Evidence Lower Bounded(ELBO) term에 대해 자세히 설명합니다.

> * 본 논문에서는 총 2가지의 Jensen inequality로 유도한 수식과 log 성질을 이용하여 유도한 수식을 소개하는데, 필자는 log 성질을 이용하여 유도한 수식만을 소개하겠습니다. (결과는 둘 다 동일합니다.)

<br>

3. The reparameterization trick : 모델을 Backpropagation 하기 위해서는 수식을 미분해야하는데, 확률이 포함된 수식에서는 미분하여 Parameters(mu, sigma)를 학습할 수 없습니다. 따라서 Monte Carlo Markov Chain(MCMC)과 학습할 Parameters와 Random성을 분리한 Trick을 이용하여 해결합니다.

<br>

4. 필자는 논문에서 소개하는 자세한 방법론 보다는 Latent Space에 대한 해석을 중점으로 설명하려합니다.

## Methods

Deep Latent Variable Model(DLVM)에 해당하는 모델 중 하나인 VAE는 다음과 같이 Log likelihood P(D|Θ)를 최대화하는 Parameter(Θ)를 찾고자 합니다.

## Code Review

1. main.py 에서의 나머지 params는 default로 학습한 결과입니다.

`python main.py --epochs 100 --log_interval 30`

<br>

2. Loss Results

- Test_dataset 기준으로 96epochs에서 다음과 같은 결과를 얻었습니다.

![Loss_results](https://user-images.githubusercontent.com/82640592/133197837-ff01fabe-0edd-4fc9-9086-edad86cc8132.jpg)

<br>

- BCE_loss는 학습을 하는 과정에서 많이 감소했으나, KLD_loss는 오히려 증가했습니다.
- 하지만 recontruction의 초점에서는 BCE_loss가 낮게 나와야 좋은 결과를 얻을 수 있습니다.
![loss](https://user-images.githubusercontent.com/82640592/133253606-18c0678c-bc30-43c1-ba88-f01f79961138.jpg)

<br>

3. X_input vs Reconstruction
 
- epoch : 1

- BCE_loss: 165.0093, KLD_loss: 5.3669, Total_loss: 170.3763


![1](https://user-images.githubusercontent.com/82640592/133198716-ffe91881-4f91-4caa-a24f-080ca9b01075.jpg)
![1](https://user-images.githubusercontent.com/82640592/133198734-05539ffe-f460-48af-b254-861f12631d6e.jpg)

- epoch : 96 (best)

- BCE_loss: 141.4159, KLD_loss: 6.5646, Total_loss: 147.9805


![96](https://user-images.githubusercontent.com/82640592/133253301-3d299682-ccd4-4d34-b7d0-16fc131eb54e.jpg)
![96](https://user-images.githubusercontent.com/82640592/133253271-9efc9479-cfa3-4118-9770-e3df4f2e6da2.jpg)

<br>

4. Latent Space Results

- 2-dim Latent space에 Isotropic한 Zero Mean Normal Distribution을 Prior Distribution으로 정의한 결과입니다.

![Latent_gif_light](https://user-images.githubusercontent.com/82640592/133202586-daa04877-208d-4d35-8fa0-b2df7ef7f2f8.gif)


위의 결과는 나름 잘(?) Clustering 되었다라고 생각할 수 있겠지만, 다음과 같은 VAE의 한계점을 볼 수 있습니다.
또한 각 한계점들을 보완하기위해 다양한 VAE architecture 기반의 방법론들이 연구되고 있습니다.

1) 학습을 하는 과정에서 Prior Distribution을 정의해 주어야 합니다.

> * 우리는 각 Dataset이 Latent space 상에서 어떤 형태의 분포를 이루는지 알 수 없습니다. 또한 안다고 하더라도 해당 pdf를 정의하기가 어렵고, KL-Divergence에 적용하여 analytic한 수식을 얻기 어렵습니다.

> * VAE는 사용하기 쉬운 Isotropic Zero Mean Normal Distribution을 Prior Distribution으로 가정하고 시작합니다. 따라서 위 그림처럼 원점(0, 0) 근처에서 원에 가까운 형태로 Clustering되는 것을 볼 수 있습니다.

> * 또한 Latent Structure의 정보를 반영하기위해 Curvature가 0(Euclidean space)이 아닌 Hyperbolic Space(음의 Curvature)나 Spherical Space(양의 Curvature) 위에서 정의한 Latent Space를 사용하는 방법론이 등장하였습니다.

>> - Hyperspherical Variational Auto-Encoders (UAI 2018)
>> - Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders (NeurIPS 2019)
>> - Mixed-curvature Variational Autoencoders (ICLR 2020)

<br>

2) Prior Distribution과 Posterior Distribution의 거리를 측정하는 Metric이 KL-divergence를 사용해야합니다.

> * "이게 왜 한계점이지?"라고 생각할 수 있으나, Deep Learning에서 두 분포간의 거리를 측정할 수 있는 Metric은 다양합니다. (예, Total Variation(TV), Jensen Shannon Divergence(JS), Wasserstein Distance 등) 또한 KL-Divergence는 두 분포가 서로 다른 영역에서 측정된 경우 완전히 다르다라는 판단을 내리게끔 설계되어 있습니다.

> * 수식과 그림으로 설명

> * 참고로 GAN의 경우 discriminator의 학습이 잘 죽게되는 원인이기도 합니다.

> * 해당 한계점을 보완하기위해 Wasserstein Auto-Encoders(ICLR 2018)과 같은 방법론이 등장하였습니다.

<br>

3) 목적에 적합한 최적의 Latent dim, Feature Extraction을 위한 Neural Network 설계에 대한 설명이 없습니다.

> * VQ-VAE, VAELLS
