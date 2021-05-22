# FaceShifter

This is my reimplement of FaceShifter - a new SOTA in FaceSwapping using Deep Neural Network

## Datasets
I planed to train on three different datasets that was reported in the paper: CelebA, FFHQ and VGGFace.

### Preprocessing
I used FFHQ's script to detect and cropping faces in the images. In VGGFace, a large number of face has very small size and low resolution, thus I decided to only keep face crop images with size larger than a constant (i.e: h,w > 96).

## Script:

### Training:
```bash
python train.py
```
### Testing:
```bash
python test.py
```
### Generate visualization:
```bash
python demo_image.py
```

## References:

https://github.com/taotaonice/FaceShifter
https://github.com/mindslab-ai/faceshifter
https://github.com/taesungp/contrastive-unpaired-translation
https://github.com/TreB1eN/InsightFace_Pytorch

```bibtex
@article{li2019faceshifter,
  title={Faceshifter: Towards high fidelity and occlusion aware face swapping},
  author={Li, Lingzhi and Bao, Jianmin and Yang, Hao and Chen, Dong and Wen, Fang},
  journal={arXiv preprint arXiv:1912.13457},
  year={2019}
}
```

