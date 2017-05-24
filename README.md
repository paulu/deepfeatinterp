# 1 Introduction

Deep Feature Interpolation (DFI) edits the content of an image by interpolating the feature representations of a deep convolutional neural network. DFI is described in [https://arxiv.org/abs/1611.05507](https://arxiv.org/abs/1611.05507) and will appear at [CVPR 2017](http://cvpr2017.thecvf.com/).

Please cite this paper if you use our work:

Paul Upchurch<sup>1</sup>, Jacob Gardner<sup>1</sup>, Geoff Pleiss, Robert Pless, Noah Snavely, Kavita Bala, Kilian Weinberger. Deep Feature Interpolation for Image Content Changes. In Computer Vision and Pattern Recognition (CVPR), 2017 

<sup>1</sup>Authors contributed equally.
<details>
  <summary>bibtex</summary>
  <pre>@inproceedings{upchurch2017deep,
  title={{D}eep {F}eature {I}nterpolation for Image Content Changes},
  author={Upchurch, Paul and Gardner, Jacob and Pleiss, Geoff and Pless, Robert and Snavely, Noah and Bala, Kavita and Weinberger, Kilian},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}</pre> 
</details>

# 2 Demonstrations

## 2.1 Demo1

![demo1](documentation/images/demo1_example.png)

This script produces six kinds of transformations (older, mouth open, eyes open, smiling, moustache, eyeglasses) on LFW faces.

```python
  python demo1.py
  # ~1.3 minutes to reconstruct each image (using 1 Titan-X)
  # Total time: 9.0 minutes
```

## 2.2 Demo2

![demo2](documentation/images/demo2_senior.jpg) ![demo2](documentation/images/demo2_man.jpg) ![demo2](documentation/images/demo2_kid.jpg)

This script ages or adds facial hair to a front-facing portrait at resolutions up to 1000x1000.

### Preparing an Images Database
This demo requires a database of high resolution images, which is used to select source and target
images for the transformation. Follow the instructions
[datasets/facemodel/README.md](datasets/facemodel/README.md) to collect the database we
use in the paper.

## 2.3 Demo3

![demo3](documentation/images/demo3_example.png)

This script fills in missing portions of shoe images.

To reconstruct one of the shoe images:
```python
  python demo3.py
  # 1.3 minutes to reconstruct each image (using 1 Titan-X)
  # Total time: 1.5 minutes
```

# 3 Options

## 3.1 Reconstruction backend (--backend)

We have two backends. Caffe+SciPy uses Caffe to forward/backward VGG
(GPU) then uses SciPy to call the FORTRAN implementation of L-BFGS-B
(CPU). Torch uses PyTorch to do the entire reconstruction on the
GPU. Torch is faster than Caffe+SciPy but it produces a lower-quality
result. We set Caffe+SciPy to be default for the LFW and UT-Zappos50K
demonstrations and Torch to be the default for the high-res face
demonstration.

## 3.2 Interpolation "amount" (--delta)

The `delta` parameter controlls how strong a transformation to make. Setting it to 0
results in no transformation at all, and larger numbers result in a stronger
transformation. You can input multiple values, like `--delta 0,0.1,0.3,0.5` to try
multiple transformations.

For most transformations, an ideal `delta` value will be between `0` and `1` (unless you use
the `--scaling none` option).

## 3.3 Other options

* `--device_id` - if you want to specify a GPU to use
* `--K` - number of nearest neighbors used to construct source/target sets
* `--iter` - number of optimization iterations for reconstruction
* `--postprocess color` - matches the color of the final image to match the original image
