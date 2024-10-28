# CEST-PSnetwork
The code of a partially separable function-based CEST reconstruction method in the article "Highly-accelerated CEST MRI using frequency-offset-dependent k-space sampling and deep-learning reconstruction".

![image](https://github.com/easyCEST/CEST-PSnetwork/blob/main/Reconstruction%20flowchart.jpg)
Reconstruction flowchart utilizing a two-step training strategy, which is based on partially separable model and deep learning. The images reconstructed from the under-sampled FOD dataset were first fed into a U-net to estimate the spectral bases, resulting in an output imageset X1 by minimization of the loss function L1. Then X1 was decomposed into spectral bases and spatial coefficients by SVD, with the decomposed two parts undergoing a 1D convolution layer and a U-net, respectively. The final reconstructed images were obtained by multiplying the output spectral part and spatial part, both with a loss function of L2.
