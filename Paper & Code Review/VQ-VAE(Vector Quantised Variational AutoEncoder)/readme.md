# Neural Discrete Representation Learning

- Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
- Published at NIPS 2017.

## Short Review

1. 본 논문은 필자가 VAE의 한계점이라고 지적했던 Prior Distribution을 정의해 주어야 한다는 부분을 Learnable한 Latent Space를 통해 해결한 연구입니다.

> * https://github.com/Giwoong-Han/Variational-Inference/tree/main/Paper%20%26%20Code%20Review/VAE(Variational%20AutoEncoder)

<br>

2. 또한 Encoder는 Continuous가 아닌 Discrete한 Code를 출력합니다.

> * Discrete한 모델은 Continuous한 모델보다 복잡한 추론, 계획 및 예측 학습에 적합하다고 알려져 있습니다.

<br>

위의 방법으로 인해 Decoder가 Encoder의 조건들을 무시하고 (Prior Fitting Term -> Loss가 0이 되는) Prior를 그대로 흉내내어 Model의 Latent Variable을 무시하는 Posterior Collapse를 해결할 수 있으며 이미지 뿐만 아니라 음성, 비디오 도메인에서도 적합한 방법론으로 알려져 있습니다.

<br>

## Methods

- D

<br>

## Code Review

1. main.py 

<br>
