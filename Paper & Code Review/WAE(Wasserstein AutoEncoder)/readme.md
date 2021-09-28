# Wasserstein Auto-Encoders

- Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, Bernhard Schoelkopf
- Published at ICLR 2018.

## Short Review

1. 필자가 VAE의 한계점이라고 지적했던 Prior Distribution과 Posterior Distribution의 Discrepancy를 측정하는 Metric의 자유도를 제시한 연구입니다.

> * https://github.com/Giwoong-Han/Variational-Inference/tree/main/Paper%20%26%20Code%20Review/VAE(Variational%20AutoEncoder)

<br>

2. VAE의 단점인 Blurry한 Samples을 만들어내는 문제점을 보완하기 위하여 GAN을 적용하는 기법들이 등장하였는데, WAE는 GAN Loss를 사용한 가장 대표적인 연구인 Adversarial Autoencoder(AAE)의 General한 방법입니다.

> * 필자는 VAE를 사용함에 있어서 제약 조건이 다양하다고 지적하였는데 그 중 Loss수식에서 해당 논문은 다음과 같은 자유도를 제시할 수 있습니다.
>> * ① 일반적으로 사용하는 L2-norm 뿐만 아닌 다양한 Cost Fuction이 사용가능합니다.
>> * ② 두 분포간의 Discrepancy를 구하는 함수를 다양하게 사용할 수 있습니다.

<br>

3. Kantorovich-Rubinstein Duality를 이용하여 Optimal Transport(OP) 문제를 두 확률분포의 Joint Distribution을 알지 못해도 적용할 수 있습니다.

> * Cost Function이 1-Lipschitz Fuction이라는 제약조건을 만족해야합니다.

## Methods

1. VAE에서의 Encoder의 문제점

> * 기존의 VAE는 2가지의 상반되는 목표를 학습하도록 설계되었습니다.

>> ① 우리가 임의로 지정한 Isotropic Zero Mean Gaussian Distribution을 Latent의 True Distribution이라 가정하고, Encoder가 Latent Distribution과 유사한 Posterior Distribution을 따르도록  학습합니다. (Prior Fitting Term)

>> ② Input에 넣었던 이미지를 잘 복원하도록 Encoder가 학습합니다. (Reconstruction Term)

> * 그러나 아래 그림처럼 인코더의 관점에서 바라보면 ①, ②번에 해당하는 Term들이 서로가 서로에게 도움을 주기 보다는 어느정도 타협을 하는 선에서 전체적인 Loss를 줄여가면서 상반되게 학습을 한다고 볼 수 있습니다. (필자의 실험에서도 전체적인 Loss는 감소하지만 KL-Divergence는 오히려 증가했습니다.)

> * WAE는 위의 단점을 어느정도 보완하여 설계되어 있습니다. 각 데이터가 개별적으로 Prior에 유사하도록 강제하는 것이 아닌 전체에 해당하는 Continuous Mixture Distribution가 유사하도록 설계하여 서로 다른 구분의 데이터들이 Latent상에서도 서로 멀리 위치하도록 학습할 수 있습니다.

![Conti](https://user-images.githubusercontent.com/82640592/135079337-fd85caa8-975c-40a2-92d6-61d7e58c62a2.jpg)

<br>

![VAE_problem](https://user-images.githubusercontent.com/82640592/135067651-e4f8947a-c8d1-46bf-aac9-93c018fdf39b.jpg)

<br>

2. Optimal Transport(OT)

> * WAE는 Coupling Theory에서 두 Probability Space가 주어졌을 때, 그 Space들을 어떤 곱의 형태로 한 번에 표현하고자하는 Probability Measure의 관점에서 시작합니다. 그 중 Optimal Transport cost에서 유도되는 다음과 같은 수식을 이용하여 저자는 새롭게 적용하였습니다.

>> ① Kantorovich's formulation

<br>

![Kantorovich_form](https://user-images.githubusercontent.com/82640592/135085524-007c00f4-40d4-40fe-81be-601f4ac9a9d8.jpg)

<br>

설명

>> ② Kantorovich-Rubinstein duality

<br>

![Kantorovich_dual](https://user-images.githubusercontent.com/82640592/135090151-cf0d6585-64fa-45d5-88a6-26832d7d47c0.jpg)

<br>

설명

>> ③ D-WAE

<br>

![thm1](https://user-images.githubusercontent.com/82640592/135090176-c23eb0f0-b0d9-4488-9ace-1082f7ccccab.jpg)

<br>

<br>

![WAE](https://user-images.githubusercontent.com/82640592/135090195-f4635d24-856a-4063-ad03-f5b60a86b920.png)

<br>

그림 설명

> -> cost function, Dz 종류 

## Code Review

1. main.py.

`python main.py`

<br>

2. Loss Results

- 
