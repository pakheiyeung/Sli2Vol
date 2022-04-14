
# Sli2Vol: Annotate a 3D Volume from a Single Slice with Self-Supervised Learning

![Figure](fig/sli2vol.gif)

This repository contains the codes (in PyTorch) for the framework introduced in the following paper:

Sli2Vol: Annotate a 3D Volume from a Single Slice with Self-Supervised Learning
[[Paper]](https://arxiv.org/abs/2105.12722) [[Project Page]](https://pakheiyeung.github.io/Sli2Vol_wp/)

```
@article{yeung2021sli2vol,
	title = {Sli2Vol: Annotate a 3D Volume from a Single Slice with Self-Supervised Learning},
	author = {Yeung, Pak-Hei and Namburete, Ana IL and Xie, Weidi},
	booktitle = {International conference on Medical Image Computing and Computer Assisted Intervention},
	pages = {69--79},
	year = {2021},
}
```
## Contents
- [Sli2Vol: Annotate a 3D Volume from a Single Slice with Self-Supervised Learning](#sli2vol-annotate-a-3d-volume-from-a-single-slice-with-self-supervised-learning)
  - [Contents](#contents)
  - [Dependencies](#dependencies)
  - [Correspondence Flow Network](#correspondence-flow-network)
  - [Verification Module](#verification-module)
   

## Dependencies
- Python (3.6), other versions should also work
- PyTorch (1.6), other versions should also work 

## Correspondence Flow Network
1. The correspondence flow network as described in the paper is coded as the *class Correspondence_Flow_Net* in `model.py`
2. It computes the affinity matrix between the input *slice1_input* and *slice2_input* and use the matrix to reconstruct *slice2_reconstructed* from the input *slice1*

## Verification Module
To do, but it would be an easy implementation based on the description in the paper.