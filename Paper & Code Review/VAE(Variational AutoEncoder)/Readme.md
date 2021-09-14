# Auto-Encoding Variational Bayes

- Diederik P Kingma, Max Welling

## Short Review

1. 본 논문은 Variational Inference, Bayesian Deep Learning을 공부할 때 가장 기본적으로 알아야 할 개념들을 소개합니다.

2. The Variational bound

> * 

3. The reparameterization trick

> * 

4. 필자는 Latent Space에 대한 해석을 중점으로 설명하려합니다.

## Methods


## Code Review

1. main.py 에서의 나머지 params는 default로 학습하였습니다.

`python main.py --epochs 100 --log_interval 30`

2. Loss Results

Test_dataset 기준으로 97epochs에서 다음과 같은 결과를 얻었습니다.

![Loss_results](https://user-images.githubusercontent.com/82640592/133197837-ff01fabe-0edd-4fc9-9086-edad86cc8132.jpg)

3. Latent Space Results

![Latent_gif_light](https://user-images.githubusercontent.com/82640592/133198505-fe134204-de8d-4d31-9934-dde48f38e8f3.gif)

