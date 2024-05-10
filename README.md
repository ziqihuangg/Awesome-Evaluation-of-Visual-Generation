# Awesome Evaluation of Visual Generation

[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fziqihuangg%2FAwesome-Evaluation-of-Visual-Generation&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

*This repository collects methods for evaluating visual generation.*

![overall_structure](./figures/fig_teaser_combined.jpg)

## Overview

### What You'll Find Here

Within this repository, we collect works that aim to answer some critical questions in the field of evaluating visual generation, such as:

- **Model Evaluation**: How does one determine the quality of a specific image or video generation model?
- **Sample/Content Evaluation**: What methods can be used to evaluate the quality of a particular generated image or video?
- **User Control Consistency Evaluation**: How to tell how well the generated images and videos align with the user controls or inputs?

### Updates

This repository is updated periodically. If you have suggestions for additional resources, updates on methodologies, or fixes for expiring links, please feel free to do any of the following:
- raise an [Issue](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation/issues),
- nominate awesome related works with [Pull Requests](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation/pulls),
- We are also contactable via email (`ZIQI002 at e dot ntu dot edu dot sg`).

### Table of Contents
- [1. Evaluation Metrics of Generative Models](#1.)
  - [1.1. Evaluation Metrics of Image Generation](#1.1.)
  - [1.2. Evaluation Metrics of Video Generation](#1.2.)
- [2. Evaluation Metrics of Condition Consistency](#2.)
  - [2.1 Evaluation Metrics of Multi-Modal Condition Consistency](#2.1.)
  - [2.2. Evaluation Metrics of Image Similarity](#2.2.)
- [3. Evaluation Systems of Generative Models](#3.)
  - [3.1. Evaluation of Text-to-Image Generation](#3.1.)
  - [3.2. Evaluation of Text-Based Image Editing](#3.2.)
  - [3.3. Evaluation of Text-to-Video Generation](#3.3.)
  - [3.4. Evaluation of Image-to-Video Generation](#3.4.)
  - [3.5. Evaluation of Model Trustworthiness](#3.5.)
- [4. Improving Visual Generation with Evaluation / Feedback / Reward](#4.)
- [5. Quality Assessment for AIGC](#5.)
- [6. Study and Rethinking](#6.)
- [7. Other Useful Resources](#7.)

<a name="1."></a>
## 1. Evaluation Metrics of Generative Models
<a name="1.1."></a>
### 1.1. Evaluation Metrics of Image Generation


| Metric | Paper | Code |
| -------- |  -------- |  ------- |
| Inception Score (IS) | [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) (NeurIPS 2016) |  |
| Fréchet Inception Distance (FID) | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (NeurIPS 2017) | [![Code](https://img.shields.io/github/stars/bioinf-jku/TTUR.svg?style=social&label=Official)](https://github.com/bioinf-jku/TTUR) [![Code](https://img.shields.io/github/stars/mseitzer/pytorch-fid.svg?style=social&label=PyTorch)](https://github.com/mseitzer/pytorch-fid) |
| Kernel Inception Distance (KID) | [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) (ICLR 2018) |   [![Code](https://img.shields.io/github/stars/toshas/torch-fidelity.svg?style=social&label=Unofficial)](https://github.com/toshas/torch-fidelity) [![Code](https://img.shields.io/github/stars/NVlabs/stylegan2-ada-pytorch.svg?style=social&label=Unofficial)](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py) 
| CLIP-FID | [The Role of ImageNet Classes in Fréchet Inception Distance](https://arxiv.org/abs/2203.06026) (ICLR 2023) | [![Code](https://img.shields.io/github/stars/kynkaat/role-of-imagenet-classes-in-fid.svg?style=social&label=Official)](https://github.com/kynkaat/role-of-imagenet-classes-in-fid)  [![Code](https://img.shields.io/github/stars/GaParmar/clean-fid.svg?style=social&label=Official)](https://github.com/GaParmar/clean-fid?tab=readme-ov-file#computing-clip-fid) |
| Precision-and-Recall | [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) (NeurIPS 2019) | [![Code](https://img.shields.io/github/stars/kynkaat/improved-precision-and-recall-metric.svg?style=social&label=OfficialTensowFlow)](https://github.com/kynkaat/improved-precision-and-recall-metric)   |
| Renyi Kernel Entropy (RKE) | [An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions](https://openreview.net/forum?id=PdZhf6PiAb) (NeurIPS 2023) | [![Code](https://img.shields.io/github/stars/mjalali/renyi-kernel-entropy.svg?style=social&label=Official)](https://github.com/mjalali/renyi-kernel-entropy)   |

<a name="1.2."></a>
### 1.2. Evaluation Metrics of Video Generation


| Metric | Paper | Code |
| -------- |  -------- |  ------- |
| FID-vid | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (NeurIPS 2017) |  |
| Fréchet Video Distance (FVD) | [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717) (arXiv 2018) | [![Code](https://img.shields.io/github/stars/songweige/TATS.svg?style=social&label=Unofficial)](https://github.com/songweige/TATS/blob/main/tats/fvd/fvd.py) |

<a name="2."></a>
## 2. Evaluation Metrics of Condition Consistency
<a name="2.1."></a>
### 2.1 Evaluation Metrics of Multi-Modal Condition Consistency


| Metric | Condition | Pipeline | Code | References | 
| -------- |  -------- |  ------- | -------- |  -------- |  
| CLIP Score (`a.k.a.` CLIPSIM) | Text | cosine similarity between the CLIP image and text embeddings |  [![Code](https://img.shields.io/github/stars/openai/CLIP.svg?style=social&label=CLIP)](https://github.com/openai/CLIP) [PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html) | [CLIP Paper](https://arxiv.org/abs/2103.00020) (ICML 2021). Metrics first used in [CLIPScore Paper](https://arxiv.org/abs/2104.08718) (arXiv 2021) and [GODIVA Paper](https://arxiv.org/abs/2104.14806) (arXiv 2021) applies it in video evaluation. |
| Mask Accuracy | Segmentation Mask | predict the segmentatio mask, and compute pixel-wise accuracy against the ground-truth segmentation mask | any segmentation method for your setting |
| DINO Similarity | Image of a Subject (human / object *etc*) | cosine similarity between the DINO embeddings of the generated image and the condition image | [![Code](https://img.shields.io/github/stars/facebookresearch/dino.svg?style=social&label=DINO)](https://github.com/facebookresearch/dino) | [DINO paper](https://arxiv.org/abs/2104.14294). Metric is proposed in [DreamBooth](https://arxiv.org/abs/2208.12242).
<!-- | Identity Consistency | Image of a Face |  | - | -->

<!-- 
Papers for CLIP Similarity:
[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (ICML 2021), [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718) (arXiv 2021), [GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions](https://arxiv.org/abs/2104.14806) (arXiv 2021) | [![Code](https://img.shields.io/github/stars/openai/CLIP.svg?style=social&label=CLIP)](https://github.com/openai/CLIP) [PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html) -->

<a name="2.2."></a>
### 2.2. Evaluation Metrics of Image Similarity

| Metrics | Paper | Code |
| -------- |  -------- |  ------- |
| Learned Perceptual Image Patch Similarity (LPIPS) | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) (2018-01-11) (CVPR 2018) | [![Code](https://img.shields.io/github/stars/richzhang/PerceptualSimilarity.svg?style=social&label=Official)](https://github.com/richzhang/PerceptualSimilarity) [![Website](https://img.shields.io/badge/Website-9cf)](https://richzhang.github.io/PerceptualSimilarity/) |
| Structural Similarity Index (SSIM) | [Image quality assessment: from error visibility to structural similarity](https://ieeexplore.ieee.org/document/1284395) (TIP 2004) |   [![Code](https://img.shields.io/github/stars/open-mmlab/mmagic.svg?style=social&label=MMEditing)](https://github.com/open-mmlab/mmagic/blob/main/tests/test_evaluation/test_metrics/test_ssim.py) [![Code](https://img.shields.io/github/stars/Po-Hsun-Su/pytorch-ssim.svg?style=social&label=Unofficial)](https://github.com/Po-Hsun-Su/pytorch-ssim) |
| Peak Signal-to-Noise Ratio (PSNR) | - |   [![Code](https://img.shields.io/github/stars/open-mmlab/mmagic.svg?style=social&label=MMEditing)](https://github.com/open-mmlab/mmagic/blob/main/tests/test_evaluation/test_metrics/test_psnr.py) |
| Multi-Scale Structural Similarity Index (MS-SSIM) | [Multiscale structural similarity for image quality assessment](https://ieeexplore.ieee.org/document/1292216) (SSC 2004) | [PyTorch-Metrics](https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html#:~:text=Compute%20MultiScaleSSIM%2C%20Multi%2Dscale%20Structural,details%20at%20different%20resolution%20scores.&text=a%20method%20to%20reduce%20metric%20score%20over%20labels.) |
| Feature Similarity Index (FSIM) | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://ieeexplore.ieee.org/document/5705575) (TIP 2011) | [![Code](https://img.shields.io/github/stars/mikhailiuk/pytorch-fsim.svg?style=social&label=Unofficial)](https://github.com/mikhailiuk/pytorch-fsim)



The community has also been using [DINO](https://arxiv.org/abs/2104.14294) or [CLIP](https://arxiv.org/abs/2103.00020) features to measure the semantic similarity of two images / frames.


There are also recent works on new methods to measure visual similarity (more will be added):

+ [DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data](https://arxiv.org/abs/2306.09344) (2023-06-15)  
  [![Code](https://img.shields.io/github/stars/ssundaram21/dreamsim.svg?style=social&label=Official)](https://github.com/ssundaram21/dreamsim)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dreamsim-nights.github.io)
  
<a name="3."></a>
## 3. Evaluation Systems of Generative Models
<a name="3.1."></a>
### 3.1. Evaluation of Text-to-Image Generation



+ [Revisiting Text-to-Image Evaluation with Gecko: On Metrics, Prompts, and Human Ratings](https://arxiv.org/abs/2404.16820) (2024-04-25)  

+ [Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation](https://arxiv.org/abs/2404.15100) (2024-04-23)  


+ [TAVGBench: Benchmarking Text to Audible-Video Generation](https://arxiv.org/abs/2404.14381) (2024-04-22)  
  [![Code](https://img.shields.io/github/stars/OpenNLPLab/TAVGBench.svg?style=social&label=Official)](https://github.com/OpenNLPLab/TAVGBench)

+ [Object-Attribute Binding in Text-to-Image Generation: Evaluation and Control](https://arxiv.org/abs/2404.13766) (2024-04-21)  

+ [Evaluating Text-to-Visual Generation with Image-to-Text Generation](https://arxiv.org/abs/2404.01291) (2024-04-01)  
  [![Code](https://img.shields.io/github/stars/linzhiqiu/t2v_metrics.svg?style=social&label=Official)](https://github.com/linzhiqiu/t2v_metrics)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://linzhiqiu.github.io/papers/vqascore)
  

+ [FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models](https://arxiv.org/abs/2403.16379) (2024-03-25)

+ [An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions](https://openreview.net/forum?id=PdZhf6PiAb) (2024-02-13)  
  [![Code](https://img.shields.io/github/stars/mjalali/renyi-kernel-entropy.svg?style=social&label=Official)](https://github.com/mjalali/renyi-kernel-entropy)

+ [Stellar: Systematic Evaluation of Human-Centric Personalized Text-to-Image Methods](https://arxiv.org/abs/2312.06116) (2023-12-11)  
  [![Code](https://img.shields.io/github/stars/stellar-gen-ai/stellar-metrics.svg?style=social&label=Official)](https://github.com/stellar-gen-ai/stellar-metrics)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://stellar-gen-ai.github.io/)


+ [A Contrastive Compositional Benchmark for Text-to-Image Synthesis: A Study with Unified Text-to-Image Fidelity Metrics](https://arxiv.org/abs/2312.02338) (2023-12-04)  
  [![Code](https://img.shields.io/github/stars/zhuxiangru/Winoground-T2I.svg?style=social&label=Official)](https://github.com/zhuxiangru/Winoground-T2I)


+ [SelfEval: Leveraging the discriminative nature of generative models for evaluation](https://arxiv.org/abs/2311.10708) (2023-11-17)


+ [GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks](https://arxiv.org/abs/2311.01361) (2023-11-02)

+ [Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation](https://arxiv.org/abs/2310.18235) (2023-10-27, ICLR 2024)  
  [![Code](https://img.shields.io/github/stars/j-min/DSG.svg?style=social&label=Official)](https://github.com/j-min/DSG)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://google.github.io/dsg/)

+ [DEsignBench: Exploring and Benchmarking DALL-E 3 for Imagining Visual Design](https://arxiv.org/abs/2310.15144) (2023-10-23)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://design-bench.github.io)
  

+ [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513) (2023-10-17)  
  [![Code](https://img.shields.io/github/stars/djghosh13/geneval.svg?style=social&label=Official)](https://github.com/djghosh13/geneval)  

+ [Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy](https://arxiv.org/abs/2310.09247) (2023-10-13)  
  [![Code](https://img.shields.io/github/stars/yandex-research/text-to-img-hypernymy.svg?style=social&label=Official)](https://github.com/yandex-research/text-to-img-hypernymy)  

+ [ImagenHub: Standardizing the evaluation of conditional image generation models](https://arxiv.org/abs/2310.01596) (2023-10-02)  
  [![Code](https://img.shields.io/github/stars/TIGER-AI-Lab/ImagenHub.svg?style=social&label=Official)](https://github.com/TIGER-AI-Lab/ImagenHub)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/ImagenHub/)
  [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)

+ [JourneyDB: A Benchmark for Generative Image Understanding](https://arxiv.org/abs/2307.00716) (2023-07-03, NeurIPS 2023)  
  [![Code](https://img.shields.io/github/stars/JourneyDB/JourneyDB.svg?style=social&label=Official)](https://github.com/JourneyDB/JourneyDB)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://journeydb.github.io/)
  

+ [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350) (2023-07)  
  [![Code](https://img.shields.io/github/stars/Karine-Huang/T2I-CompBench.svg?style=social&label=Official)](https://github.com/Karine-Huang/T2I-CompBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://karine-h.github.io/T2I-CompBench/)
  

+ [Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341) (2023-06)  
  [![Code](https://img.shields.io/github/stars/tgxs002/HPSv2.svg?style=social&label=Official)](https://github.com/tgxs002/HPSv2)


+ [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569) (2023-05)  
  [![Code](https://img.shields.io/github/stars/yuvalkirstain/PickScore.svg?style=social&label=Official)](https://github.com/yuvalkirstain/PickScore)


+ [Human Preference Score: Better Aligning Text-to-Image Models with Human Preference](https://arxiv.org/abs/2303.14420) (2023-03-25)  
  [![Code](https://img.shields.io/github/stars/tgxs002/align_sd.svg?style=social&label=Official)](https://github.com/tgxs002/align_sd)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tgxs002.github.io/align_sd_web/)

+ [Benchmarking Spatial Relationships in Text-to-Image Generation](https://arxiv.org/abs/2212.10015) (2022-12-20)  
  [![Code](https://img.shields.io/github/stars/microsoft/VISOR.svg?style=social&label=Official)](https://github.com/microsoft/VISOR)


<a name="3.2."></a>
### 3.2. Evaluation of Text-Based Image Editing


+ [EditVal: Benchmarking Diffusion Based Text-Guided Image Editing Methods](https://arxiv.org/abs/2310.02426) (2023-10-03)  
  [![Code](https://img.shields.io/github/stars/deep-ml-research/editval_code.svg?style=social&label=Official)](https://github.com/deep-ml-research/editval_code)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://deep-ml-research.github.io/editval/)
  
+ [Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting](https://arxiv.org/abs/2212.06909) (2022-12-13, CVPR 2023)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://research.google/blog/imagen-editor-and-editbench-advancing-and-evaluating-text-guided-image-inpainting/)


<a name="3.3."></a>
### 3.3. Evaluation of Text-to-Video Generation


+ [Subjective-Aligned Dataset and Metric for Text-to-Video Quality Assessment](https://arxiv.org/abs/2403.11956) (2024-03-18)  
  [![Code](https://img.shields.io/github/stars/QMME/T2VQA.svg?style=social&label=Official)](https://github.com/QMME/T2VQA)



+ [Sora Generates Videos with Stunning Geometrical Consistency](https://arxiv.org/abs/2402.17403) (2024-02-27)  
  [![Code](https://img.shields.io/github/stars/meteorshowers/Sora-Generates-Videos-with-Stunning-Geometrical-Consistency.svg?style=social&label=Official)](https://github.com/meteorshowers/Sora-Generates-Videos-with-Stunning-Geometrical-Consistency)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sora-geometrical-consistency.github.io)


+ [STREAM: Spatio-TempoRal Evaluation and Analysis Metric for Video Generative Models](https://arxiv.org/abs/2403.09669) (2024-01-30)  
  [![Code](https://img.shields.io/github/stars/pro2nit/STREAM.svg?style=social&label=Official)](https://github.com/pro2nit/STREAM)


+ [Towards A Better Metric for Text-to-Video Generation](https://arxiv.org/abs/2401.07781) (2024-01-15)  
  [![Code](https://img.shields.io/github/stars/showlab/T2VScore.svg?style=social&label=Official)](https://github.com/showlab/T2VScore)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/T2VScore/)

+ [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) (2023-11-29)  
  [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VBench-project/)

+ [FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation](https://arxiv.org/abs/2311.01813) (2023-11-03)  
  [![Code](https://img.shields.io/github/stars/llyx97/FETV.svg?style=social&label=Official)](https://github.com/llyx97/FETV)

+ [EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://arxiv.org/abs/2310.11440) (2023-10-17)  
  [![Code](https://img.shields.io/github/stars/EvalCrafter/EvalCrafter.svg?style=social&label=Official)](https://github.com/EvalCrafter/EvalCrafter)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://evalcrafter.github.io)

+ [StoryBench: A Multifaceted Benchmark for Continuous Story Visualization](https://arxiv.org/abs/2308.11606) (2023-08-22, NeurIPS 2023)  
  [![Code](https://img.shields.io/github/stars/google/storybench.svg?style=social&label=Official)](https://github.com/google/storybench)

<a name="3.4."></a>
### 3.4. Evaluation of Image-to-Video Generation


+ I2V-Bench from [ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation](https://arxiv.org/abs/2402.04324) (2024-02-06)  
  [![Code](https://img.shields.io/github/stars/TIGER-AI-Lab/ConsistI2V.svg?style=social&label=Official)](https://github.com/TIGER-AI-Lab/ConsistI2V)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/ConsistI2V/)

+ [AIGCBench: Comprehensive Evaluation of Image-to-Video Content Generated by AI](https://arxiv.org/abs/2401.01651) (2024-01-03)  
  [![Code](https://img.shields.io/github/stars/BenchCouncil/AIGCBench.svg?style=social&label=Official)](https://github.com/BenchCouncil/AIGCBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.benchcouncil.org/AIGCBench/)

+ [VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) (2024-03) from [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) (2023-11-29)  
  [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VBench-project/)

<a name="3.5."></a>
### 3.5. Evaluation of Model Trustworthiness

#### 3.5.1. Evaluation of Visual-Generation-Model Trustworthiness


+ [VBench-Trustworthiness](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_trustworthiness) (2024-03) from [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) (2023-11-29)  
  [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VBench-project/)


+ [Holistic Evaluation of Text-To-Image Models](https://arxiv.org/abs/2311.04287) (2023-11-07)  
  [![Code](https://img.shields.io/github/stars/stanford-crfm/helm.svg?style=social&label=Official)](https://github.com/stanford-crfm/helm)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://crfm.stanford.edu/helm/heim/v1.1.0/)

#### 3.5.2. Evaluation of Non-Visual-Generation-Model Trustworthiness
Not for visual generation, but related evaluations of other models like LLMs

+ [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249) (2024-02-06)  
  [![Code](https://img.shields.io/github/stars/centerforaisafety/HarmBench.svg?style=social&label=Official)](https://github.com/centerforaisafety/HarmBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.harmbench.org)


<a name="4."></a>
## 4. Improving Visual Generation with Evaluation / Feedback / Reward

+ [Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2405.00760) (2024-05-01)  

+ [ID-Aligner: Enhancing Identity-Preserving Text-to-Image Generation with Reward Feedback Learning](https://arxiv.org/abs/2404.15449) (2024-04-23)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://idaligner.github.io)
  [![Code](https://img.shields.io/github/stars/Weifeng-Chen/ID-Aligner.svg?style=social&label=Official)](https://github.com/Weifeng-Chen/ID-Aligner)


+ [ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](https://arxiv.org/abs/2404.07987) (2024-04-11)  

+ [UniFL: Improve Stable Diffusion via Unified Feedback Learning](https://arxiv.org/abs/2404.05595) (2024-04-08)  

+ [ByteEdit: Boost, Comply and Accelerate Generative Image Editing](https://arxiv.org/abs/2404.04860) (2024-04-07)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://byte-edit.github.io/)

+ [Aligning Diffusion Models by Optimizing Human Utility](https://arxiv.org/abs/2404.04465) (2024-04-06)  

+ [Confidence-aware Reward Optimization for Fine-tuning Text-to-Image Models](https://arxiv.org/abs/2404.01863) (2023-04-02, ICLR 2024)  

+ [VersaT2I: Improving Text-to-Image Models with Versatile Reward](https://arxiv.org/abs/2403.18493) (2024-03-27)  

+ [RL for Consistency Models: Faster Reward Guided Text-to-Image Generation](https://arxiv.org/abs/2404.03673) (2024-03-25)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://rlcm.owenoertell.com)
  [![Code](https://img.shields.io/github/stars/Owen-Oertell/rlcm.svg?style=social&label=Official)](https://github.com/Owen-Oertell/rlcm)

+ [Rich Human Feedback for Text-to-Image Generation](https://arxiv.org/abs/2312.10240) (2023-12-15, CVPR 2024)  

+ [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://arxiv.org/abs/2312.12490) (2023-12-19)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://instructvideo.github.io)

+ [DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback](https://arxiv.org/abs/2311.17946) (2023-11-29)  

+ [Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908) (2023-11-21)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://blog.salesforceairesearch.com/diffusion-dpo/)
  [![Code](https://img.shields.io/github/stars/SalesforceAIResearch/DiffusionDPO.svg?style=social&label=Official)](https://github.com/SalesforceAIResearch/DiffusionDPO)
  [![Code](https://img.shields.io/github/stars/huggingface/diffusers.svg?style=social&label=diffusers)](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/diffusion_dpo)

+ [Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) (2023-10-05)  
  [![Code](https://img.shields.io/github/stars/mihirp1998/AlignProp.svg?style=social&label=Official)](https://github.com/mihirp1998/AlignProp/)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://align-prop.github.io/)

+ [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) (2023-09-29)  

+ [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) (2023-04-12)  
  [![Code](https://img.shields.io/github/stars/THUDM/ImageReward.svg?style=social&label=Official)](https://github.com/THUDM/ImageReward)

+ [Human Preference Score: Better Aligning Text-to-Image Models with Human Preference](https://arxiv.org/abs/2303.14420) (2023-03-25)  
  [![Code](https://img.shields.io/github/stars/tgxs002/align_sd.svg?style=social&label=Official)](https://github.com/tgxs002/align_sd)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tgxs002.github.io/align_sd_web/)

<!-- ## Evaluation Datasets
- UCF101
- ImageNet
- COCO -->

<a name="5."></a>
## 5. Quality Assessment for AIGC

### 5.1. Image Quality Assessment for AIGC

+ [AIGIQA-20K: A Large Database for AI-Generated Image Quality Assessment](https://arxiv.org/abs/2404.03407) (2024-04)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image/summary)
  

+ [Exploring the Naturalness of AI-Generated Images](https://arxiv.org/abs/2312.05476) (2023-12-09)  
  [![Code](https://img.shields.io/github/stars/zijianchen98/AGIN.svg?style=social&label=Official)](https://github.com/zijianchen98/AGIN)
  


+ [AIGCIQA2023: A Large-scale Image Quality Assessment Database for AI Generated Images: from the Perspectives of Quality, Authenticity and Correspondence](https://arxiv.org/abs/2307.00211) (2023-07-01)  
  
### 5.2. Aesthetic Predictors for Generated Images

+ Aesthetic Scorer extension for SD Automatic WebUI (2023-01-15)  
  [![Code](https://img.shields.io/github/stars/vladmandic/sd-extension-aesthetic-scorer.svg?style=social&label=Official)](https://github.com/vladmandic/sd-extension-aesthetic-scorer)


+ Simulacra Aesthetic-Models (2022-07-09)  
  [![Code](https://img.shields.io/github/stars/crowsonkb/simulacra-aesthetic-models.svg?style=social&label=Official)](https://github.com/crowsonkb/simulacra-aesthetic-models)


+ LAION-Aesthetics_Predictor V2: CLIP+MLP Aesthetic Score Predictor (2022-06-26)  
  [![Code](https://img.shields.io/github/stars/christophschuhmann/improved-aesthetic-predictor.svg?style=social&label=Official)](https://github.com/christophschuhmann/improved-aesthetic-predictor)
  [![Website](https://img.shields.io/badge/Visualizer-9cf)](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://laion.ai/blog/laion-aesthetics/#laion-aesthetics-v2)


+ LAION-Aesthetics_Predictor V1 (2022-05-21)  
  [![Code](https://img.shields.io/github/stars/LAION-AI/aesthetic-predictor.svg?style=social&label=Official)](https://github.com/LAION-AI/aesthetic-predictor)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://laion.ai/blog/laion-aesthetics/#laion-aesthetics-v1)


<!-- ## Video Quality Assessment for AIGC
- To be added -->

<a name="6."></a>
## 6. Study and Rethinking

### 6.1. Evaluation of Evaluations
+ [Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt Coherence Metrics with T2IScoreScore (TS2)](https://arxiv.org/abs/2404.04251) (2024-04)  
  [![Code](https://img.shields.io/github/stars/michaelsaxon/T2IScoreScore.svg?style=social&label=Official)](https://github.com/michaelsaxon/T2IScoreScore)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://t2iscorescore.github.io)

### 6.2. Survey

+ [Evaluating Text-to-Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) (2024-03-18)  


### 6.3. Study
+ [On the Content Bias in Fréchet Video Distance](https://arxiv.org/abs/2404.12391) (2024-04-18, CVPR 2024)  
  [![Code](https://img.shields.io/github/stars/songweige/content-debiased-fvd.svg?style=social&label=Official)](https://github.com/songweige/content-debiased-fvd)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://content-debiased-fvd.github.io)

+ [A Study on the Evaluation of Generative Models](https://arxiv.org/abs/2206.10935) (2022-06)

+ [A Note on the Inception Score](https://arxiv.org/abs/1801.01973) (2018-01)

### 6.4. Competition
+ [NTIRE 2024 Quality Assessment of AI-Generated Content Challenge](https://arxiv.org/abs/2404.16687) (2024-04-25)

<a name="7."></a>
## 7. Other Useful Resources
 + Stanford Course: CS236 "Deep Generative Models" - Lecture 15 "Evaluation of Generative Models" [[slides]](https://deepgenerativemodels.github.io/assets/slides/lecture15.pdf)

<!-- 
Papers to read and to organize:
- Rethinking FID: Towards a Better Evaluation Metric for Image Generation 
- Wasserstein Distortion: Unifying Fidelity and Realism
-->
