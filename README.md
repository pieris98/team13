# team13
Team 13 repository for [ML3D](https://mhsung.github.io/kaist-cs479-fall-2023/) course project @KAIST
## Introduction
This is our project "Geometric Deep Learning for 3D Shape Correspondence".
It explores several recent methods on 3D Shape correspondence, including: 
- CNN methods with local operators e.g. Geodesic CNN [1] and MoNet[2]
- methods evaluating functional maps on geometry e.g. Geometric Functional Maps [3] and
- our main focus (baseline method), [DiffusionNet](https://github.com/nmwsharp/diffusion-net/tree/master)[4].
## Code and Data
The README in each folder explains the prerequisite development environment, code and data needed.
## Evaluation
Our code is evaluated on the [FAUST Humans dataset](https://faust-leaderboard.is.tuebingen.mpg.de/)[5], a standard realistic benchmark for 3D Shape Correspondence methods.
## References
[1] Masci, J. et al. 2015. Geodesic Convolutional Neural Networks on Riemannian Manifolds. 2015 IEEE International Conference on Computer Vision Workshop (ICCVW) (Dec. 2015)
[2] Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, & Michael M. Bronstein. (2016). Geometric deep learning on graphs and manifolds using mixture model CNNs.
[3] Nicolas Donati, Abhishek Sharma, & Maks Ovsjanikov. (2020). Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence. 
[4] Sharp, N. et al. 2022. DiffusionNet: Discretization Agnostic Learning on Surfaces. arXiv.
[5] Bogo, F. et al. 2014. FAUST: Dataset and Evaluation for 3D Mesh Registration. 2014 IEEE Conference on Computer Vision and Pattern Recognition (Columbus, OH, USA, Jun. 2014)

