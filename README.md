# Deep learning

## Repository with Pytorch/Keras/numpy implementation of multiple machine learning algorithm/ Projects

### Currently :

Supervised Learning :

  - Implementation of ResNet34 and ResNet18 + model trained on cifar-10
    linked article : https://arxiv.org/pdf/1512.03385.pdf
    
  - Implementation of a convolutionnal neural network with numpy
  
  ** TODO : **
    - Add ResNe50 and +
    - Add vectorizaion of computation for convolution (numpy)
      linked article : https://arxiv.org/pdf/1501.07338.pdf
    - Add BatchNormalization layer and transpose convolution (numpy)
    - Add numpy implementation in C++
    - Add numpy implementation in VHDL on a FPGA
    
Unsupervised Learning :

  - Implementation of Auto-Encoder and VAE (Keras)
    linked article : https://arxiv.org/pdf/1312.6114.pdf

  - Implementation of VanillaGAN 
    linked article : https://arxiv.org/pdf/1406.2661.pdf

  - Implementation of DCGAN 
    linked article : https://arxiv.org/pdf/1511.06434.pdf

  - Implementation of WGAN-GP
    linked article : https://arxiv.org/pdf/1704.00028.pdf
  
  ** TODO : **
    - Add FCC-GAN 
      linked article : https://arxiv.org/pdf/1905.02417.pdf
    - Add Adversarial VAE
      linked article : https://arxiv.org/pdf/1701.04722.pdf
    - Generate new pokemon with VAE and FCC-GAN
    - Generate mask for thresholding with Auto Encoder

Reinforcement Learning:

  - Implementation of Proximal policy optimisation + model trained on openAI gym Bipedal-Walker and pendulum
    linked article : https://arxiv.org/pdf/1707.06347.pdf

  - Implementation of REINFORCE (Vanilla Policy Gradient method)

  - Implementation of actor-critic

  ** TODO : **
    - Fix DDPG : https://arxiv.org/pdf/1509.02971.pdf
    - Add DQN  : https://arxiv.org/pdf/1312.5602.pdf
    - Train an agent on tetris with DQN
    - Add GridWorld with Tabular Q-Learning