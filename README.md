# Enhancing Adversarial Transferability with Partial Blocks on Vision Transformer

Yanyang Han, Ju Liu, Xiaoxi Liu, Xiao Jiang, Lingchen Gu, Xuesong Gao, Weiqiang Chen

**[code link](https://github.com/yanyanghan/PBSA.git)** 

![Partial Blocks Search Process](f2.pdf)
![Adversarial Attack Process](f3.pdf)

> **Abstract:** 
*Adversarial examples can attack multiple unknown convolutional neural networks (CNNs) due to adversarial transferability, which reveals the vulnerability of CNNs and facilitates the development of adversarial attacks. However, most of the existing adversarial attack methods possess a limited transferability on vision transformers (ViTs). In this paper, we propose a partial blocks search attack (PBSA) method to generate adversarial examples on ViTs, which significantly enhance transferability. Instead of directly employing the same strategy for all encoder blocks on ViTs, we divide encoder blocks into two categories by introducing the block weight score and exploit distinct strategies to process them. In addition, we optimize the generation of perturbations by regularizing the self-attention feature maps and creating an ensemble of partial blocks. Finally, perturbations are adjusted by an adaptive weight to disturb the most effective pixels of original images. Extensive experiments on the ImageNet dataset are conducted to demonstrate the validity and effectiveness of the proposed PBSA. The experimental results reveal the superiority of the proposed PBSA to state-of-the-art attack methods on both ViTs and CNNs. Furthermore, PBSA can be flexibly combined with existing methods, which significantly enhances the transferability of adversarial examples.*

## Contents
1) [Requirements](#Requirements)
2) [Dataset](#Dataset)
3) [Partial Blocks Search Attack](#Partial Blocks Search Attack)
4) [References](#references)
5) [Citation](#citation)

## Requirements
```bash
pip install -r requirements.txt
```

## Dataset
We select 1000 images of different categories from the ILSVRC 2012 validation dataset. In our experiments, all these images are resized to 224 × 224 × 3 before being fed into ViTs to load model parameters correctly.

## Partial Blocks Search Attack
<sup>([top](#contents))</sup>
 `DATA_DIR` points to the root directory containing the validation images of ImageNet (original imagenet). We support attack types FGSM, PGD, MI-FGSM, DIM, TI, and SGM by default. 

```bash
python test.py \
  --test_dir "$DATA_DIR" \
  --src_model deit_tiny_patch16_224 \
  --tar_model tnt_s_patch16_224  \
  --attack_type pma \
  --eps 16 \
  --index "all" \
  --batch_size 128
```
For other model families, the pretrained models will have to be downloaded and the paths updated in the relevant files under `vit_models`.

## References
<sup>([top](#contents))</sup>
Code borrowed from [DeiT](https://github.com/facebookresearch/deit) repository, [TRM](https://github.com/Muzammal-Naseer/On-Improving-Adversarial-Transferability-of-Vision-Transformers.git) repository and [TIMM](https://github.com/rwightman/pytorch-image-models) library. We thank them for their wonderful code bases. 

## Citation
If you find our work useful, please consider giving a star :star: and citation.
```bibtex
@misc{yyh,
      title={Enhancing Adversarial Transferability with Partial Blocks on Vision Transformer}, 
      author={Yanyang Han and Ju Liu and Xiaoxi Liu and Xiao Jiang and Lingchen Gu and Xuesong Gao and Weiqiang Chen},
      year={2022}
}
```
