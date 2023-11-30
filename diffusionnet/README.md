# [Original Implementation of DiffusionNet repository](https://github.com/nmwsharp/diffusion-net/tree/master)
**DiffusionNet** is a general-purpose method for deep learning on surfaces such as 3D triangle meshes and point clouds. It is well-suited for tasks like segmentation, classification, feature extraction, etc.
## 

## Code structure Outline
  - `experiments/functional_correspondence` scripts to reproduce experiments from our version of DiffusionNet for 3D shape correspondence on FAUST dataset.
  - `environment.yml` A conda environment file which can be used to install packages.


## Prerequisites
### conda environment
DiffusionNet depends on pytorch, as well as a handful of other fairly typical numerical packages. These can usually be installed manually without much trouble, but alternately a conda environment file is also provided (see conda documentation for additional instructions). These package versions were tested with CUDA 10.1 and 11.1. 

```
conda env create --name diffusion_net -f environment.yml
```
The code assumes a GPU with CUDA support. DiffusionNet has minimal memory requirements; >4GB GPU memory should be sufficient. 
4. 

### Required Packages & Data
Besides the conda environment, to run the code you will need to:
1. Install `pyshot`, a python bindings library to be able to compute SHOT descriptors. Follow our forked [pyshot repo](https://github.com/pieris98/pyshot) for instructions on how to install `pyshot` and its _dependencies_.
2. Download the [FAUST-remeshed](https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx) (`FAUST_r.zip` and `FAUST_r_vts.zip`) datasets and extract them under `experiments/functional_correspondence/data/faust/off_2/` and `experiments/functional_correspondence/data/faust/corres/` respectively.
3. Download our precomputed LPS features [here](https://drive.google.com/file/d/15bGZiKCvKDNjyaFyvsyLUZ7taZM7eIY5/view?usp=sharing) (we computed them on FAUST_r dataset using MATLAB code from [this repo](https://github.com/yiqun-wang/LPS-matlab/tree/master)). Move the `lps.pt` file to `experiments/functional_correspondence/data/faust/lps.pt`

### Tips and Tricks

By default, DiffusionNet uses _spectral acceleration_ for fast performance, which requires some CPU-based precomputation to compute operators & eigendecompositions for each input, which can take a few seconds for moderately sized inputs. DiffusionNet will be fastest if this precomputation only needs to be performed once for the dataset, rather than for each input. 

- If you are learning on a **template mesh**, consider precomputing operators for the _reference pose_ of the template, but then using xyz the coordinates of the _deformed pose_ as inputs to the network. This is a slight approximation, but will make DiffusionNet very fast, since the precomputed operators are shared among all poses.
- If  you need **data augmentation**, try to apply augmentations _after_ computing operators whenever possible. For instance, in our examples, we apply random rotation to positions, but only _after_ computing operators. Note that we find common augmentations such as slightly skewing/scaling/subsampling inputs are generally unnecessary with DiffusionNet.
