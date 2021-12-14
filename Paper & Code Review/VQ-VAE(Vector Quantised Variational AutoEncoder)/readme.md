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

> * 즉, D-dim긴 K개의 Vector 중 Encoder의 Output과 가장 Euclidean Distance가 가장 가까운 k를 찾아 나중에 Decoder의 인풋으로 사용할 Zq(x)라 정의합니다.

<br>

2. Model

![t](https://user-images.githubusercontent.com/82640592/145817713-2ca74f04-72d6-4db9-bcda-fbbedc311996.jpg)

> * 기존의 VAE에서는 encoder의 아웃풋을 그대로 Latent Space로 사용합니다. 저자는 Learnable한 Latent space를 만들기 위하여 위와 같이 Ze(x)와 Embedding space를 사용하였습니다. Embedding Space는 K만큼의 Size를 가지는 Discrete한 Latent Space로 구성되어 각 D-dimension만큼의 크기로 설정되어 있습니다. 여기서 초기 Embedding Space의 값들은 Gaussian Distribution이 아닌 Uniform Distribution을 따릅니다. 그 이유로 저자는 인코더의 Parameter의 설정과 유사하게 설정하면 ELBO의 KL-divergence term의 값이 항상 Constant로 나와 학습과정에서 무시할 수 있다고 말합니다.

> * Forward Path에서는 먼저 Encoder를 통해서 나온 Ze(x)와 앞서 1.에서 정의한 Euclidean distance를 기반으로 가장 유사한 e를 찾습니다. 위의 그림은 Embedding Space의 2번째 Latent가 가장 유사하다는 가정이 전제되어 있습니다. 그 이후 Decoder에 가장 유사했던 e=Zq(x)를 input으로 Reconstruction을 진행합니다. Forward Process가 종료되고 Decoder에서부터 순차적으로 Gradient를 구한 후 흘려주는 Backward를 진행합니다. 여기서 주목해야할 점은 해당 Forward Path를 통해 얻은 Gradient가 다시 Embedding Space의 e에 흘려주는 것이 아니라 Encoder의 Output인 Ze(x)에 전달해 준다는 것입니다. 즉, 이 방법을 통해 Prior Fitting이 아닌 Encoder가 Reconstruction이 잘 되도록 유용한 정보만 줄 수 있어 기존의 VAE의 한계점이었던 Posterior Collapse를 해결할 수 있다고 말합니다.

<br>

3. Loss Function

![1](https://user-images.githubusercontent.com/82640592/145993065-dd04b847-76e9-4759-a8f6-e76ae3b0b229.jpg)

<br>

> * 위와 같이 3개의 Term으로 Loss Fuction을 정의합니다. 위의 수식에서 Sg(Stop gradient)는 Forward시 Patial Derivatives가 0으로 학습에 반영되지 않습니다.

>> ① Reconstruction Term

>> * Embedding Space에서 뽑은 Zq(x)가 주어졌을때 Decoder를 통해 얻은 Output Reconstruction이 잘 되도록 제약하는 Term입니다. Encoder는 해당 Term에 관여하지 않습니다.

<br>

>> ② Codebook Term

>> * Backward에 반영되는 부분은 e로 Encoder에서 얻은 Output Ze(x)와 Euclidean Distance가 더 가까워지도록 합니다.

<br>

>> ③ Commitmet Loss

>> * 학습에 사용가능한 Parameter 중 Embedding Space의 K에 대한 부분에 대해 생각해보면, 값이 너무 커지게되면 Embedding Space가 K배로 증가하여 학습에 시간이 매우 오래걸릴 수 있습니다. 따라서 저자는 위와 같은 Commitment Loss를 통하여 Embedding Space와 Encoder의 Output이 유사해지도록 설계하였습니다. 여기서 Beta는 0.1~2.0까지 변경해도 Robust한 결과를 얻을 수 있고 저자는 0.25를 실험에 사용하였습니다.

## Code Review

1. main.py 

추후 업데이트 하겠습니다.

<br>
