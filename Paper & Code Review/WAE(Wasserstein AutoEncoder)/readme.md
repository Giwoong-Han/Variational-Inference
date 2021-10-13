# Wasserstein Auto-Encoders

- Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, Bernhard Schoelkopf
- Published at ICLR 2018.

## Short Review

1. 필자가 VAE의 한계점이라고 지적했던 Prior Distribution과 Posterior Distribution의 Discrepancy를 측정하는 Metric의 자유도를 제시한 연구입니다.

> * https://github.com/Giwoong-Han/Variational-Inference/tree/main/Paper%20%26%20Code%20Review/VAE(Variational%20AutoEncoder)

<br>

2. VAE의 단점인 Blurry한 Samples을 만들어내는 문제점을 보완하기 위하여 GAN을 적용하는 기법들이 등장하였는데, WAE는 GAN Loss를 사용한 가장 대표적인 연구인 Adversarial Autoencoder(AAE)의 General한 방법입니다.

> * 필자는 VAE를 사용함에 있어서 제약 조건이 다양하다고 지적하였는데 그 중 Loss수식에서 해당 논문은 다음과 같은 자유도를 제시할 수 있습니다.
>> * ① 두 분포간의 Discrepancy를 구하는 함수를 다양하게 사용할 수 있습니다. (저자는 Generative Adverserial Networks(GAN)기반과 Maximum Mean Discrepancy(MMD)기반을 사용했습니다.)
>> * ② 다양한 Kernel 함수를 이용하여 MMD loss에 적용할 수 있다. (저자는 RBF(Gaussian)와 Inverse Multi Quadric(IMQ)를 제안합니다.)
>> * ③ Prior 분포를 Gaussian Distribution만 사용할 필요가 없다.

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

>> ① Kantorovich's Formulation

<br>

![Kantorovich_form](https://user-images.githubusercontent.com/82640592/135085524-007c00f4-40d4-40fe-81be-601f4ac9a9d8.jpg)

<br>

> * 위의 수식은 두 개의 Random Variable이 만드는 모든 Joint Distribution Space에서 어떤 Cost Function을 줄이는 문제를 풀고자 합니다.

<br>

>> ② Kantorovich-Rubinstein Duality

<br>

![Kantorovich_dual](https://user-images.githubusercontent.com/82640592/135090151-cf0d6585-64fa-45d5-88a6-26832d7d47c0.jpg)

<br>

> * 그 중에서 위와 같은 Duality를 만족하면 두 분포의 Joint Distribution을 구하는 어려운 문제를 풀지 않아도 해결할 수 있습니다.

<br>

>> ③ D-WAE

<br>

![thm1](https://user-images.githubusercontent.com/82640592/135090176-c23eb0f0-b0d9-4488-9ace-1082f7ccccab.jpg)

<br>

> * ②번의 Duality덕분에 ①번의 Kantorovich's Formulation을 보다 다루기 쉬운 Form으로 변형할 수 있습니다. 이는 VAE의 관점에서 바라보면 동일하게 Prior Fitting Term과 Reconstruction Term이 포함된 수식으로 볼 수 있습니다.

<br>

![WAE](https://user-images.githubusercontent.com/82640592/135090195-f4635d24-856a-4063-ad03-f5b60a86b920.png)

<br>

> * 저자는 Numerical한 Solution을 얻을 수 있도록 Qz와 Pz가 유사해지는 Term을(마치 VAE의 Prior Fitting Term처럼) 강제로 제약시켜 위와 같은 최종 수식으로 WAE의 목적함수를 정의합니다. 논문에서 사용한 Cost Fuction은 L2-norm을 사용하였고, Dz(Qz,Pz) term을 다음과 같이 두 가지의 방법으로 해결하고자 합니다.

<br>

3. MMD-based & GAN-based

<br>

> * ① GAN-based

<br>

>> * GAN에서 주로 사용하는 Jensen-Shannon divergence를 이용하여 Discriminator를 학습합니다.

<br>

![Dr](https://user-images.githubusercontent.com/82640592/136030861-422851b6-01e6-45a9-8bd5-d936f6216a67.jpg)

<br>

>> * Disciminator와 Encoder, Decoder를 서로 번갈아 학습하고 저자는 Disciminator의 lr를 Decoder의 절반으로 설정하여 Overfitting을 방지하였습니다.

<br>

> * ② MMD-based

<br>

![MMD](https://user-images.githubusercontent.com/82640592/136028714-80158bb8-ba11-4941-b352-643229e3c66f.jpg)

<br>

>> * 해당 내용을 논문에서는 위와 같이 Reproducing Kernel Hilbert Space(RKHS)에서 적분과 각각의 Marginal Probability를 이용하여 기술하였습니다.

<br>

>> * 모르는 두 분포의 Discrepancy를 구하는 가장 쉬운 방법은 서로의 Expectation을 비교하는 것 일겁니다. 하지만 Expectation만 비교하기에는 두 분포가 가진 특성을 잘 비교했다고 말하기는 어렵습니다. 따라서 각 분포에서 파생되는 Moment들을 모두 비교하기 위해서 Kernel을 이용한 Maximum Mean Discrepancy(MMD)을 구하고자 합니다. 다음의 Gaussian Kernel의 예시를 보시면 Kernel이 두 분포로 부터 얻은 Moment의 차의 합으로 표현됨을 알 수 있습니다.

<br>

![moment](https://user-images.githubusercontent.com/82640592/136174235-554e1cf0-0a63-4dc5-b871-12c46b712244.jpg)

<br>

>> * ②-1  Inverse Multi Quadric(IMQ) Kernel

<br>

![IMQ](https://user-images.githubusercontent.com/82640592/136028721-7a3b3318-3b49-4ec3-b879-45a851121784.jpg)

<br>

>>> * 논문에서 MNIST 실험에 대하여 Setup한 수식입니다.

<br>

>> * ②-2 Gaussian(RBF) Kernel

<br>

>>> * 논문에서는 IMQ Kernel만 사용하였는데, 그 이유는 Gaussian(RBF) Kernel이 Quick Tail Decay Problem이 있어서 Outlier에 더 많은 패널티를 부여하지 않도록 보다 Heavier Tail을 가지는 IMQ Kernel만 사용하였습니다. (두 분포를 그려보면 꼬리 부분이 IMQ가 훨씬 두껍습니다.)

## Code Review

1. main.py.

`python main.py -batch_size 128 -n_z 2 -Pz normal[sphere] -Dz GAN[MMD]`

<br>

2. Loss Results

> ① MMD & Gaussian

![loss_normal_MMD](https://user-images.githubusercontent.com/82640592/136145789-62c59cbd-20a1-4225-a0c7-ddd5f00ba6e3.jpg)

> ② MMD & Sphere

![loss_sphere_MMD](https://user-images.githubusercontent.com/82640592/136145796-994cbdf1-d8c6-4dee-b644-09e43496bd24.jpg)

> ③ GAN & Gaussian

![loss_normal_GAN](https://user-images.githubusercontent.com/82640592/136145802-7978a6bd-a688-4e0f-ab7c-a71dcb16d6c7.jpg)

> ④ GAN & Sphere

>> 9epoch 이후 Gradient Exploding이 발생하였습니다.

<br>

3. X_input vs Reconstruction

> ① MMD & Gaussian

>> epoch : 1

>> REC_loss: 5.1177, MMD_loss: 38.4637, Total_loss: 43.5814

![1](https://user-images.githubusercontent.com/82640592/136035014-f34f2d12-1f27-4e3f-9c40-8af2802e8a90.jpg)
![1](https://user-images.githubusercontent.com/82640592/136034926-1f5da177-098a-4f34-a273-08570fa62484.jpg)


<br>

>> epoch : 100

>> REC_loss: 4.0441, MMD_loss: 0.6215, Total_loss: 4.6656

![100](https://user-images.githubusercontent.com/82640592/136144980-ac9242de-ac2c-4b7e-add1-8252389cbfad.jpg)
![100](https://user-images.githubusercontent.com/82640592/136144992-a59f066d-9904-458f-9943-72915f59c0c4.jpg)

<br>

> ② MMD & Sphere

>> epoch : 1

>> REC_loss: 5.2891, MMD_loss: 21.9796, Total_loss: 27.2687

![1](https://user-images.githubusercontent.com/82640592/136035508-20487ba6-f5a9-43e1-87b4-59864f60086c.jpg)
![1](https://user-images.githubusercontent.com/82640592/136035598-71643ea2-b195-4cdc-b2e1-e6d9d0720f45.jpg)

<br>

>> epoch : 100

>> REC_loss: 4.3482, MMD_loss: 1.3615, Total_loss: 5.7096
 
![100](https://user-images.githubusercontent.com/82640592/136145119-b92ca560-f422-4047-9bd8-8d8ae6f7ce44.jpg)
![100](https://user-images.githubusercontent.com/82640592/136145109-ca31ab89-ac5e-42a4-b9d7-12781ad96ae0.jpg)

<br>

> ③ GAN & Gaussian

>> epoch : 1

>> REC_loss: 6.3179, Dz_loss: 270.8663, Total_loss: 277.1842

![1](https://user-images.githubusercontent.com/82640592/136036544-bcd67a59-572d-4969-b011-39baf6e8f3e4.jpg)
![1](https://user-images.githubusercontent.com/82640592/136036324-d712aee0-30eb-4657-a5ca-0f8bbd5e67e2.jpg)


<br>

>> epoch : 100

>> REC_loss: 5.0200, Dz_loss: 265.0048, Total_loss: 270.0248

![100](https://user-images.githubusercontent.com/82640592/136145391-38519c0b-40e4-4397-8f0c-3f026a477fd0.jpg)
![100](https://user-images.githubusercontent.com/82640592/136145370-44899545-f01f-4d23-848b-79f037f14b5d.jpg)

<br>

4. Latent Space Results

> ① MMD & Gaussian

![normal_MMD](https://user-images.githubusercontent.com/82640592/136150763-48e6ea86-5c67-490b-abee-549f82127dde.gif)

> ② MMD & Sphere

![sphere_MMD](https://user-images.githubusercontent.com/82640592/136150827-8634ddf5-60c9-42fd-88a1-4a1371358bde.gif)

> ③ GAN & Gaussian

![normal_GAN](https://user-images.githubusercontent.com/82640592/136150855-4ab973c4-fbd9-4f45-bf1a-47026120182c.gif)

<br>

5. CIFAR10

> Latent Dimension을 64로 설정하고 실험한 결과입니다.

> ① MMD & Gaussian

>> REC_loss: 12.2893, MMD_loss: 2.1517, Total_loss: 14.4410

![1](https://user-images.githubusercontent.com/82640592/137117189-6d95ff9b-bac9-4732-bb11-2bdad427831a.jpg)
![1](https://user-images.githubusercontent.com/82640592/137117193-90650dd0-c447-4fc9-bdd6-310f4d1148db.jpg)

<br>

![100](https://user-images.githubusercontent.com/82640592/137117207-3cc0ccc2-5944-4d83-bf3e-4b5a9559b35d.jpg)
![100](https://user-images.githubusercontent.com/82640592/137117226-f74d59b1-1e0f-4884-8612-5c5758bd2e0c.jpg)

<br>

> ② MMD & Sphere

>> REC_loss: 12.6358, MMD_loss: 4.0119, Total_loss: 16.6477

![1](https://user-images.githubusercontent.com/82640592/137117273-ee1b57d2-a3d0-4d23-951e-76618a6e070a.jpg)
![1](https://user-images.githubusercontent.com/82640592/137117288-db3b155a-bb11-43f8-908d-10a7301627d4.jpg)

<br>

![100](https://user-images.githubusercontent.com/82640592/137117319-cc1c20f3-0333-4b5c-bfa5-3ef50c504d05.jpg)
![100](https://user-images.githubusercontent.com/82640592/137117346-7a63c5fd-2673-41c5-a65c-ac8b074aec5c.jpg)

> ③ GAN & Gaussian

>> REC_loss: 14.4658, Dz_loss: 266.0779, Total_loss: 280.5437

![1](https://user-images.githubusercontent.com/82640592/137117564-fb24e304-7cbc-4f98-8ca8-6451de740c8b.jpg)
![1](https://user-images.githubusercontent.com/82640592/137117571-ca9588d4-6f1a-4d9b-a56b-d24f14512b64.jpg)

<br>

![100](https://user-images.githubusercontent.com/82640592/137117577-bbb50755-06da-4019-9af7-0dae2ead0121.jpg)
![100](https://user-images.githubusercontent.com/82640592/137117587-b7690d34-1525-4787-adcd-3a3bbda4afc7.jpg)

<br>

> ④ GAN & Sphere

>> REC_loss: 14.4397, Dz_loss: 201.1274, Total_loss: 215.5670

![1](https://user-images.githubusercontent.com/82640592/137118312-7e90e1ac-c867-40f0-9cbe-15c3a7948e0f.jpg)
![1](https://user-images.githubusercontent.com/82640592/137118323-02507263-0b23-49dd-b130-074be4c5cdd7.jpg)

<br>

![53](https://user-images.githubusercontent.com/82640592/137118279-77d2d2e0-a844-48f8-add2-77dfc1b822a2.jpg)
![53](https://user-images.githubusercontent.com/82640592/137118289-9a850bed-f53f-4d27-b7c6-1892c731f812.jpg)
