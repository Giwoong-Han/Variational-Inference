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


3. X_input vs Reconstruction
 
- epoch : 1

- BCE_loss: 165.0093, KLD_loss: 5.3669, Total_loss: 170.3763


![1](https://user-images.githubusercontent.com/82640592/133198716-ffe91881-4f91-4caa-a24f-080ca9b01075.jpg)
![1](https://user-images.githubusercontent.com/82640592/133198734-05539ffe-f460-48af-b254-861f12631d6e.jpg)

- epoch : 97 (best)

- BCE_loss: 141.7242, KLD_loss: 6.6025, Total_loss: 148.3267


![97](https://user-images.githubusercontent.com/82640592/133198834-1d42376c-4267-4846-9dba-438a8b48b0cf.jpg)
![97](https://user-images.githubusercontent.com/82640592/133198820-bc1bcbc9-74ae-4081-99f4-6728dded5107.jpg)



4. Latent Space Results

![Latent_gif_light](https://user-images.githubusercontent.com/82640592/133198505-fe134204-de8d-4d31-9934-dde48f38e8f3.gif)

