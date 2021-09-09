# Variational Autoencoder with Learned Latent Structure

- Marissa C. Connor, Gregory H. Canal, Christopher J. Rozell
- Accepted at The 24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021)

## Short Review

1. VAE를 공부하다보면 Prior Distribution에 대한 가정이 필요하다는 문제점이 존재합니다. 하지만 일반적으로 사용하는 Gaussian을 따른다는 가정이 Image Domain에서는 센 가정이 될 수 있다고 생각합니다. 따라서 해당 문제를 해결하기 위하여, 많은 사람들이 다양한 접근을 합니다. 필자 또한 마찬가지로 해당 문제를 다양하게 바라보아야 한다는 관점을 가지고 있으며, Prior로 사용하는 Latent Space의 가정을 하나씩 줄여나가는 것을 생각하고 있습니다. (필자는 우선 Eculidean space를 깨는것에 초점을 두고 있습니다.)

> * 저자는 Learnable한 Generative Manifold Model을 제안하여 위의 문제를 해결했습니다.


2. VAE는 일반적으로 Linear한 Layers를 통해 정보를 압축하고 Euclidean Latent Space를 거쳐 정보를 복원하려합니다.

> * 저자는 위와 같은 방법이 True Data Manifold에서 벗어날 수 있다고 지적합니다. 


3. VAE는 모든 Data들이 Latent Space의 원점 주위에 대부분 군집화 되어있는 문제가 발생합니다. 따라서 Latent Space에서 데이터의 클래스 분리가 적절히 되지 않으면, Latent Space를 통과한 이후 복원된 Data가 과연 잘 클래스를 분류할 수 있도록 복원되었을지 의문이 듭니다. (필자가 관심있어하는 Unsupervised Anomaly Task에서도 해당 문제가 중요하게 작용할 것이라 생각합니다.)

> * 저자 또한 해당 문제를 지적하여 Latent Space의 적절한 Manifold 학습의 중요성을 강조합니다.

## Methods
