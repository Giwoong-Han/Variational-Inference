# Neural Discrete Representation Learning

- Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
- Published at NIPS 2017.

## Short Review

1. 본 논문은 필자가 VAE의 한계점이라고 지적했던 Prior Distribution을 정의해 주어야 한다는 부분을 Learnable한 Latent Space를 통해 해결한 연구입니다.

> * https://github.com/Giwoong-Han/Variational-Inference/tree/main/Paper%20%26%20Code%20Review/VAE(Variational%20AutoEncoder)

<br>

2. 또한 Encoder는 Continuous가 아닌 Discrete한 Code를 출력합니다.

> * Log-Likelihood를 사용하는 Discrete한 모델은 Continuous한 모델보다 더 좋은 성능을 준다고 알려져 있습니다. 

<br>

위의 방법으로 인해 Decoder가 Encoder의 조건들을 무시하고 (Prior Fitting Term -> Loss가 0이 되는) Prior를 그대로 흉내내어 Model의 Latent Variable을 무시하는 Posterior Collapse를 해결할 수 있으며 이미지 뿐만 아니라 음성, 비디오 도메인에서도 적합한 방법론으로 알려져 있습니다.

## Methods

1. Discrete Latent Variable

> * 저자는 Discrete하고 Learnable한 latent를 만들기위해 KxD 사이즈의 Embedding space를 도입하였습니다. Embedding space의 각 Vector들은 Uniform distribution을 따르며 이는 Encoder 파라미터와 동일하게 설정되기 위함입니다.

> * Encoder를 통해서 출력되는 Ze(x)는 D-dimension의 형태라고 가정하면 저자는 Prior로 설정한 KxD의 Embedding space에서 Ze(x)와 Euclidean distance가 가장 가까운 e하나를 다음과 같은 수식을 통해 구합니다. 나머지는 현재 학습에서 Forward, Backward 모두 반영되지 않습니다.

![Eu](https://user-images.githubusercontent.com/82640592/145713333-65e167e1-2510-48bf-b8da-a12480dcfe07.jpg)



2. Model

<br>

## Code Review

1. main.py 

<br>
