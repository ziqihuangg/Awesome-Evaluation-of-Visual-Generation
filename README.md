# Awesome Evaluation of Visual Generation

[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fziqihuangg%2FAwesome-Evaluation-of-Visual-Generation&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

*This repository collects methods for evaluating visual generation.*


#### What You'll Find Here

Within this repository, we collect works that aim to answer some critical questions in the field of visual generation evaluation, such as:

- **Model Evaluation**: How good is a particular image / video generation model? How does one determine the quality of a specific image or video generation model?
- **Sample/Content Evaluation**: How good is a particular generated image / video?  What metrics or methods can be used to evaluate the quality of a particular generated image or video?
- **User Control Consistency**: Are the generated images and videos aligning with the user controls or inputs?

#### Updates

This repository is updated periodically. If you have suggestions for additional resources, updates on methodologies, or fixes for expiring links, please feel free to reach out. For suggestions, feedback, or pointing out issues, please use the GitHub Issues tab. We are also contactable via email (`ZIQI002 at e dot ntu dot edu dot sg`).

## Evaluation Metrics of Generative Models

### Evaluation Metrics of Image Generation


| Metrics | Paper | Code |
| -------- |  -------- |  ------- |
| Inception Score (IS) | [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) (NeurIPS 2016) |  |
| Fréchet Inception Distance (FID) | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (NeurIPS 2017) | [![Code](https://img.shields.io/github/stars/bioinf-jku/TTUR.svg?style=social&label=Official)](https://github.com/bioinf-jku/TTUR) [![Code](https://img.shields.io/github/stars/mseitzer/pytorch-fid.svg?style=social&label=PyTorch)](https://github.com/mseitzer/pytorch-fid) |
| Kernel Inception Distance (KID) | [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401) (ICLR 2018) |   [![Code](https://img.shields.io/github/stars/toshas/torch-fidelity.svg?style=social&label=Unofficial)](https://github.com/toshas/torch-fidelity) [![Code](https://img.shields.io/github/stars/NVlabs/stylegan2-ada-pytorch.svg?style=social&label=Unofficial)](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py) 
| CLIP-FID | [The Role of ImageNet Classes in Fréchet Inception Distance](https://arxiv.org/abs/2203.06026) (ICLR 2023) | [![Code](https://img.shields.io/github/stars/kynkaat/role-of-imagenet-classes-in-fid.svg?style=social&label=Official)](https://github.com/kynkaat/role-of-imagenet-classes-in-fid)  [![Code](https://img.shields.io/github/stars/GaParmar/clean-fid.svg?style=social&label=Official)](https://github.com/GaParmar/clean-fid?tab=readme-ov-file#computing-clip-fid) |
| Precision-and-Recall | [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991) (NeurIPS 2019) | [![Code](https://img.shields.io/github/stars/kynkaat/improved-precision-and-recall-metric.svg?style=social&label=OfficialTensowFlow)](https://github.com/kynkaat/improved-precision-and-recall-metric)   |


### Evaluation Metrics of Video Generation


| Metrics | Paper | Code |
| -------- |  -------- |  ------- |
| FID-vid | [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (NeurIPS 2017) |  |
| Fréchet Video Distance (FVD) | [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717) (arXiv 2018) | [![Code](https://img.shields.io/github/stars/songweige/TATS.svg?style=social&label=Unofficial)](https://github.com/songweige/TATS/blob/main/tats/fvd/fvd.py) |


## Evaluation Metrics of Condition Consistency


| Metrics | Condition | Pipeline | Code | References | 
| -------- |  -------- |  ------- | -------- |  -------- |  
| CLIP Score (`a.k.a.` CLIPSIM) | Text | cosine similarity between the CLIP image and text embeddings |  [![Code](https://img.shields.io/github/stars/openai/CLIP.svg?style=social&label=CLIP)](https://github.com/openai/CLIP) [PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html) | [CLIP Paper](https://arxiv.org/abs/2103.00020) (ICML 2021). Metrics first used in [CLIPScore Paper](https://arxiv.org/abs/2104.08718) (arXiv 2021) and [GODIVA Paper](https://arxiv.org/abs/2104.14806) (arXiv 2021) applies it in video evaluation. |
| Mask Accuracy | Segmentation Mask | predict the segmentatio mask, and compute pixel-wise accuracy against the ground-truth segmentation mask | any segmentation method for your setting |
| DINO Similarity | Image of a Subject (human / object *etc*) | cosine similarity between the DINO embeddings of the generated image and the condition image | [![Code](https://img.shields.io/github/stars/facebookresearch/dino.svg?style=social&label=DINO)](https://github.com/facebookresearch/dino) | [DINO paper](https://arxiv.org/abs/2104.14294). Metric is proposed in [DreamBooth](https://arxiv.org/abs/2208.12242).
<!-- | Identity Consistency | Image of a Face |  | - | -->

<!-- 
Papers for CLIP Similarity:
[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (ICML 2021), [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718) (arXiv 2021), [GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions](https://arxiv.org/abs/2104.14806) (arXiv 2021) | [![Code](https://img.shields.io/github/stars/openai/CLIP.svg?style=social&label=CLIP)](https://github.com/openai/CLIP) [PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html) -->



## Evaluation Systems of Generative Models

### Evaluation of Text-to-Image


+ [Evaluating Text-to-Visual Generation with Image-to-Text Generation](https://arxiv.org/abs/2404.01291) (2024-04)  
  [![Code](https://img.shields.io/github/stars/linzhiqiu/t2v_metrics.svg?style=social&label=Official)](https://github.com/linzhiqiu/t2v_metrics)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://linzhiqiu.github.io/papers/vqascore)
  

+ [FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models](https://arxiv.org/abs/2403.16379) (2024-03-25)


+ [SelfEval: Leveraging the discriminative nature of generative models for evaluation](https://arxiv.org/abs/2311.10708) (2023-11-17)


+ [GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks](https://arxiv.org/abs/2311.01361) (2023-11-02)

+ [ImagenHub: Standardizing the evaluation of conditional image generation models](https://arxiv.org/abs/2310.01596) (2023-10-02)  
  [![Code](https://img.shields.io/github/stars/TIGER-AI-Lab/ImagenHub.svg?style=social&label=Official)](https://github.com/TIGER-AI-Lab/ImagenHub)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/ImagenHub/)
  [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)

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


### Evaluation of Text-to-Video


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

### Evaluation of Image-to-Video


+ I2V-Bench from [ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation](https://arxiv.org/abs/2402.04324) (2024-02-06)  
  [![Code](https://img.shields.io/github/stars/TIGER-AI-Lab/ConsistI2V.svg?style=social&label=Official)](https://github.com/TIGER-AI-Lab/ConsistI2V)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/ConsistI2V/)

+ [AIGCBench: Comprehensive Evaluation of Image-to-Video Content Generated by AI](https://arxiv.org/abs/2401.01651) (2024-01-03)  
  [![Code](https://img.shields.io/github/stars/BenchCouncil/AIGCBench.svg?style=social&label=Official)](https://github.com/BenchCouncil/AIGCBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.benchcouncil.org/AIGCBench/)

+ [VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) (2024-03) from [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) (2023-11-29)  
  [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VBench-project/)


### Evaluation of Model Trustworthiness

#### Evaluation of Visual-Generation-Model Trustworthiness


+ [VBench-Trustworthiness](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_reliability) (2024-03) from [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982) (2023-11-29)  
  [![Code](https://img.shields.io/github/stars/Vchitect/VBench.svg?style=social&label=Official)](https://github.com/Vchitect/VBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/VBench-project/)


+ [Holistic Evaluation of Text-To-Image Models](https://arxiv.org/abs/2311.04287) (2023-11-07)  
  [![Code](https://img.shields.io/github/stars/stanford-crfm/helm.svg?style=social&label=Official)](https://github.com/stanford-crfm/helm)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://crfm.stanford.edu/helm/heim/v1.1.0/)

#### Evaluation of Non-Visual-Generation-Model Trustworthiness
Not for visual generation, but related evaluations of other models like LLMs

+ [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249) (2024-02-06)  
  [![Code](https://img.shields.io/github/stars/centerforaisafety/HarmBench.svg?style=social&label=Official)](https://github.com/centerforaisafety/HarmBench)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.harmbench.org)



## Improving Visual Generation with Evaluation / Feedback / Reward


+ [InstructVideo: Instructing Video Diffusion Models with Human Feedback](https://arxiv.org/abs/2312.12490) (2023-12-19)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://instructvideo.github.io)

+ [DreamSync: Aligning Text-to-Image Generation with Image Understanding Feedback](https://arxiv.org/abs/2311.17946) (2023-11-29)  


+ [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) (2023-04-12)  
  [![Code](https://img.shields.io/github/stars/THUDM/ImageReward.svg?style=social&label=Official)](https://github.com/THUDM/ImageReward)

+ [Human Preference Score: Better Aligning Text-to-Image Models with Human Preference](https://arxiv.org/abs/2303.14420) (2023-03-25)  
  [![Code](https://img.shields.io/github/stars/tgxs002/align_sd.svg?style=social&label=Official)](https://github.com/tgxs002/align_sd)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tgxs002.github.io/align_sd_web/)

<!-- ## Evaluation Datasets
- UCF101
- ImageNet
- COCO -->

## Quality Assessment for AIGC

### Image Quality Assessment for AIGC

+ [AIGIQA-20K: A Large Database for AI-Generated Image Quality Assessment](https://arxiv.org/abs/2404.03407) (2024-04)  
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.modelscope.cn/datasets/lcysyzxdxc/AIGCQA-30K-Image/summary)
  

+ [Exploring the Naturalness of AI-Generated Images](https://arxiv.org/abs/2312.05476) (2023-12-09)  
  [![Code](https://img.shields.io/github/stars/zijianchen98/AGIN.svg?style=social&label=Official)](https://github.com/zijianchen98/AGIN)
  


+ [AIGCIQA2023: A Large-scale Image Quality Assessment Database for AI Generated Images: from the Perspectives of Quality, Authenticity and Correspondence](https://arxiv.org/abs/2307.00211) (2023-07-01)  
  

<!-- ## Video Quality Assessment for AIGC
- To be added -->

## Study and Rethinking

### Evaluation of Evaluations
+ [Who Evaluates the Evaluations? Objectively Scoring Text-to-Image Prompt Coherence Metrics with T2IScoreScore (TS2)](https://arxiv.org/abs/2404.04251) (2024-04)  
  [![Code](https://img.shields.io/github/stars/michaelsaxon/T2IScoreScore.svg?style=social&label=Official)](https://github.com/michaelsaxon/T2IScoreScore)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://t2iscorescore.github.io)

### Survey

+ [Evaluating Text-to-Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) (2024-03-18)  


### Study

+ [A Study on the Evaluation of Generative Models](https://arxiv.org/abs/2206.10935) (2022-06)

+ [A Note on the Inception Score](https://arxiv.org/abs/1801.01973) (2018-01)


## Useful Resources
 + Stanford Course: CS236 "Deep Generative Models" - Lecture 15 "Evaluation of Generative Models" [[slides]](https://deepgenerativemodels.github.io/assets/slides/lecture15.pdf)
