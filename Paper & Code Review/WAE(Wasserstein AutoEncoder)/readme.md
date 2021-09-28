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

> * 2가지의 상충되는 목표를 달성하도록 학습 (두 가지 설명 추가)

2. Optimal Transport(OT)

> 1) Kantorovich's formulation

> 2) Kantorovich-Rubinstein duality

> 3) D-WAE -> cost function, Dz 종류 

## Code Review

1. main.py 에서의 나머지 params는 default로 학습한 결과입니다.

`python main.py --epochs 100 --log_interval 30`

<br>

2. Loss Results

- Test_dataset 기준으로 96epochs에서 다음과 같은 결과를 얻었습니다.
