

# Gaussian Mixture Model Convolutional Networks

This is a Pytorch implementation of Gaussian Mixture Model Convolutional Networks (MoNet) for the task dense intrinsic shape correspondence, as described in the paper:

Monti *et al*, [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402) (CVPR 2017)

Pre-trained `.pth` files are provided as well as `eval.txt` showing training/test losses and accuracies on query mesh correspondences with the reference mesh.

## Requirements
* [Pytorch](https://pytorch.org/) (1.3.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (1.3.0)

## MoNet

MoNet uses a local system of pseudo-coordinates <img src="svgs/9284e17b2f479e052a85e111d9f17ce1.svg" align=middle width=21.178245pt height=14.55728999999999pt/> around to represent the neighborhood <img src="svgs/1276e542ca3d1d00fd30f0383afb5d08.svg" align=middle width=34.239315pt height=24.56552999999997pt/> and a family of learnable weighting functions w.r.t. <img src="svgs/129c5b884ff47d80be4d6261a476e9f1.svg" align=middle width=10.462980000000003pt height=14.55728999999999pt/>, e.g., Gaussian kernels <img src="svgs/f1cee86600f26eed52126ed72d2dfdd8.svg" align=middle width=305.181195pt height=37.803480000000015pt/> with learnable mean <img src="svgs/e0eef981c0301bb88a01a36ec17cfd0c.svg" align=middle width=17.106870000000004pt height=14.102549999999994pt/> and covariance <img src="svgs/aff3fd40bc3e8b5ce3ad3f61175cb17a.svg" align=middle width=20.84082pt height=22.473000000000006pt/>. The convolution is
<p align="center"><img src="svgs/1c07d8ffda7593d98eda6d17de7db825.svg" align=middle width=202.08705pt height=51.658694999999994pt/></p>

where <img src="svgs/6fccf0465699020081a15631f4a45ae1.svg" align=middle width=8.143030500000002pt height=22.745910000000016pt/> is the learnable filter weights and <img src="svgs/796df3d6b2c0926fcde961fd14b100e7.svg" align=middle width=16.021665000000002pt height=14.55728999999999pt/> is the node feature vector.

We provide efficient Pytorch implementation of this operator ``GMMConv``, which is accessible from ``Pytorch Geometric``.

## Dense Shape Correspondence

```
python -m correspondence.main
```


## Data

In order to use your own dataset, you can simply create a regular python list holding `torch_geometric.data.Data` objects and specify the following attributes:

- ``data.x``: Node feature matrix with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: Graph connectivity in COO format with shape ``[2, num_edges]`` and type ``torch.long``
- ``data.edge_attr``: Pesudo-coordinates with shape ``[num_edges, pesudo-coordinates-dim]``
- ``data.y``: Target to train against


## Cite

Please cite [this paper](https://arxiv.org/abs/1611.08402) if you use this code in your own work:

```
@inproceedings{monti2017geometric,
  title={Geometric deep learning on graphs and manifolds using mixture model cnns},
  author={Monti, Federico and Boscaini, Davide and Masci, Jonathan and Rodola, Emanuele and Svoboda, Jan and Bronstein, Michael M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5115--5124},
  year={2017}
}
```
