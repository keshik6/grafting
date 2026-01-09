<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Exploring Diffusion Transformer Designs via Grafting</h1>      
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan&nbsp;Chandrasegaran</a><sup>*1,2</sup>,&nbsp;
    <a href="https://zymrael.github.io" target="_blank" style="text-decoration: none;">Michael&nbsp;Poli</a><sup>*1,2</sup>,
              <a href="https://danfu.org" target="_blank" style="text-decoration: none;">Daniel&nbsp;Y.&nbsp;Fu</a><sup>3,4</sup>,
    <a href="https://sites.google.com/view/dongjun-kim"  target="_blank">Dongjun&nbsp;Kim</a><sup>1</sup>,<br/>
    <a href="https://lea-m-hadzic.github.io" target="_blank">Lea&nbsp;M.&nbsp;Hadzic</a><sup>1</sup>,
    <a href="https://limanling.github.io" target="_blank">Manling&nbsp;Li</a><sup>1,5</sup>,
    <a href="https://web.stanford.edu/~agrim/" target="_blank">Agrim&nbsp;Gupta</a><sup>6</sup>,
    <a href="https://www.linkedin.com/in/stefano-massaroli-b49ba8130/"  target="_blank">Stefano&nbsp;Massaroli</a><sup>2,7</sup>,<br>
     <a href="http://azaliamirhoseini.com"  target="_blank">&nbsp;Azalia Mirhoseini</a><sup>1</sup>,
    <a href="https://www.niebles.net"  target="_blank">&nbsp;Juan Carlos Niebles</a><sup>&dagger;1,8</sup>,
    <a href="https://cs.stanford.edu/~ermon/"  target="_blank">&nbsp;Stefano Ermon</a><sup>&dagger;1</sup>,
    <a href="https://profiles.stanford.edu/fei-fei-li"  target="_blank">&nbsp;Li Fei-Fei</a><sup>&dagger;1</sup><a><br/>
<span class="author-block"><sup>1</sup>&nbsp;Stanford University&nbsp;&nbsp;</span>
<span class="author-block"><sup>2</sup>&nbsp;Liquid AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>3</sup>&nbsp;Together AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>4</sup>&nbsp;UC San Diego&nbsp;&nbsp;</span><br/>
<span class="author-block"><sup>5</sup>&nbsp;Northwestern University&nbsp;&nbsp;</span>
<span class="author-block"><sup>6</sup>&nbsp;Google DeepMind&nbsp;&nbsp;</span>
<span class="author-block"><sup>7</sup>&nbsp;RIKEN&nbsp;&nbsp;</span>
<span class="author-block"><sup>8</sup>&nbsp;Salesforce Research&nbsp;&nbsp;</span><br/>
<sup>*</sup>&nbsp;Equal contribution, <sup>&dagger;</sup>&nbsp;Equal senior authorship<br/>
NeurIPS 2025 Oral<br/>
<a href="https://grafting.stanford.edu" title="Website" target="_blank" rel="nofollow" style="text-decoration: none;">🌎Website</a> |
<a href="https://huggingface.co/grafting/" title="Grafted Models" target="_blank" rel="nofollow" style="text-decoration: none;">🤗 Grafted Models</a> |
<a href="https://arxiv.org/abs/2506.05340" title="arXiv" target="_blank" rel="nofollow" style="text-decoration: none;">📄 arXiv</a> |
<a href="https://www.liquid.ai/research/exploring-diffusion-transformer-designs-via-grafting" title="Blog" target="_blank" rel="nofollow" style="text-decoration: none;">✍️ Blog</a>
</p>


![teaser_fig](https://github.com/user-attachments/assets/be81e026-877e-4c31-85e9-2cfbb81c9016)


## 📣 News

- **[2026-01-07]: Training/ Evaluation code released**
- **[2025-06-10]: Grafting codebase released**


## Abstract
Designing model architectures requires decisions such as selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width). However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation. Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models? To this end, we present grafting, a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets. Informed by our analysis of activation behavior and attention locality, we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality. Using this testbed, we develop a family of hybrid designs via grafting: replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants. Notably, many hybrid designs achieve good quality (FID: 2.38-2.64 vs. 2.27 for DiT-XL/2) using <2% pretraining compute. We then graft a text-to-image model (PixArt-Sigma), achieving a 1.43x speedup with less than a 2% drop in GenEval score. Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting. This reduces model depth by 2x and yields better quality (FID: 2.77) than other models of comparable depth. Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring.


## About this code
The Grafting codebase is written in Pytorch and provides a simple implementation for grafting Diffusion Transformers (DiTs).

## Grafted models
We provide 22 grafted models for ImageNet-1K 256×256 generation.

| Operator | Replacement Operator | Grafting Ratio | FID  | Download Link |
|----------|----------------------|----------------|------|------|
| MLP      | MLP (Self-grafting, r=4)              | 100%           | 2.54 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_4-100p-fid2.54) |
| MLP      | MLP (r=3)              | 50%            | 2.53 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_3-50p-fid2.53) |
| MLP      | MLP (r=3)              | 75%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_3-75p-fid2.61) |
| MLP      | MLP (r=6)              | 50%            | 2.38 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_6-50p-fid2.38) |
| MLP      | MLP (r=6)              | 75%            | 2.37 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_6-75p-fid2.37) |
| MLP      | Hyena-X (r=2)              | 50%            | 2.64 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-hyena_x-50p-fid2.64) |
| MLP      | Hyena-X (r=2)            | 75%            | 3.26 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-hyena_x-75p-fid3.26) |
| MHA      | MHA (Self-grafting)                  | 100%           | 2.49 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mha-100p-fid2.49) |
| MHA      | Hyena-SE             | 50%            | 2.73 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.73) |
| MHA      | Hyena-SE             | 50%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.73) |
| MHA      | Hyena-SE             | 75%            | 3.62 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.61_ablation) |
| MHA      | Hyena-X              | 50%            | 2.74 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-50p-fid2.74) |
| MHA      | Hyena-X              | 50%            | 2.58 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-50p-fid2.58_ablation) |
| MHA      | Hyena-X              | 75%            | 3.69 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-75p-fid3.69) |
| MHA      | Hyena-Y              | 50%            | 2.72 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-50p-fid2.72) |
| MHA      | Hyena-Y              | 50%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-50p-fid2.61_ablation) |
| MHA      | Hyena-Y              | 75%            | 3.66 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-75p-fid3.66) |
| MHA      | SWA                  | 50%            | 2.67 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-50p-fid2.67) |
| MHA      | SWA                  | 50%            | 2.62 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-50p-fid2.62_ablation) |
| MHA      | SWA                  | 75%            | 3.09 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-75p-fid3.09) |
| MHA      | Mamba-2              | 50%            | 2.65 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mamba_2-50p-fid2.65) |
| MHA      | Mamba-2              | 75%            | 3.02 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mamba_2-75p-fid3.02) |

## Getting Started
Start generating samples using our grafted models (See `demo_notebooks/grafting_demo.ipynb`)

##  Training Pipeline for Grafting Diffusion Transformers

This guide describes the complete training pipeline for grafting on the ImageNet-1K dataset. The pipeline is modular and can be adapted to different operators, layers and resolutions as needed. All the results reported in the paper can be reproduced using this codebase. All experiments are specified via YAML config files. We provide Dockerfiles. **For reference, we provide a step-by-step demo for replacing 3 Multi-Head Attention (MHA) operators in DiT-XL/2 with Hyena-Y operator**:
1. Data preparation & feature extraction
2. Stage 1: Activation distillation
3. Stage 2: Lightweight fine-tuning
4. Sampling + FID evaluation

---

### 1) Data Preparation & Feature Extraction

#### 1.1 Setup Environment
- Build Docker image: `docker build -t grafting .`
- (Optional) Create a persistent cache volume for downloading Hugging Face models:
  `docker volume create huggingface_cache`

- Run container (An example shown below):
  
  `docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ~/keshik/workspace/projects/grafting:/workspace -v huggingface_cache:/home/user/.cache/huggingface -v ~/keshik/data:/data -it grafting /bin/bash`

---

#### 1.2 Extract VAE Latents (Full ImageNet-1K)
- Download ImageNet-1K dataset from [here](https://www.image-net.org/download.php). 

- Extract SD-VAE features for the ImageNet-1K dataset at 256×256:
  
  `bash bash_scripts/imagenet_1k/extract_vae_fts.sh`

- Expected output directory created: `/data/vae_features/imagenet_256/train/`

- Generates a stratified 128k ImageNet-1K subset (10% used in the paper) and saves image paths + SHA hash so the exact subset can be used across different experiments. This can be increased up to the full ImageNet size if required.
  
  `bash bash_scripts/dit_imagenet_1k_256x256/generate_dataset_hash.sh`

⚡Recommended 1× H100

**Note:** Extracted SD-VAE features for ImageNet-1K dataset can be downloaded from Hugging Face: [sd_vae_features_imagenet_1k_256x256](https://huggingface.co/datasets/grafting/sd_vae_features_imagenet_1k_256x256).

---

#### 1.3 Extract DiT Block Activations (for Activation Distillation)
- Stage-1 requires intermediate DiT-XL/2 activations:
  
  `bash bash_scripts/dit_imagenet_1k_256x256/extract_mha_scion_fts.sh`

- Inside the script, users must manually set:

  - `SPLIT=train` for the training set

  - `SPLIT=val` for the validation set


- This produces:

  - `/data/scion_fts_mha/train/`

  - `/data/scion_fts_mha/val/`


⚡Recommended 1× H100

---

### 2) Grafting Stage 1: Activation Distillation

- Train replacement attention/MLP operators by distilling the extracted activations:
  
  `bash bash_scripts/dit_imagenet_1k_256x256/train_stage1.sh`

- Stage-1 trained operator checkpoints are saved under: `./results/`

- Optional post Stage-1 sampling:
  
  `bash bash_scripts/dit_imagenet_1k_256x256/sample_stage1.sh`

⚡Recommended 1× H100 (You can run this in parallel for different layers)

---

### 3) Grafting Stage 2: Lightweight Fine-Tuning

- Perform end-to-end fine-tuning after activation distillation:
  
  `bash bash_scripts/dit_imagenet_1k_256x256/train_stage2.sh`

- Stage-1 trained operator checkpoints are saved under: `./results/`

⚡Recommended 8× H100

---

### 4) Sampling & FID Evaluation

- Generate samples from the fine-tuned model and save as `.npz`:
  
  `bash bash_scripts/dit_imagenet_1k_256x256/sample_stage2.sh`

- Then compute FID using OpenAI’s reference batch.Frist, install dependencies using the official [`requirements.txt`](https://github.com/openai/guided-diffusion/blob/main/evaluations/requirements.txt), or use the Dockerfile at `assets/tf_Dockerfile/Dockerfile` for evaluation. Then run the following:

  `cd ./external/guided_diffusion/evaluations/ && wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz && python evaluator.py VIRTUAL_imagenet256_labeled.npz ./samples/demo/hyena_y_6_16_27.npz`

⚡Recommended 8× H100

---


## Contact

- Keshigeyan Chandrasegaran: keshik@stanford.edu
- Michael Poli: poli@stanford.edu

For issues, feedback, or contributions, please open an issue or submit a pull request.

## Acknowledgements

We acknowledge the following works and libraries:

- Scalable Diffusion Models with Transformers (DiT): https://github.com/facebookresearch/DiT
- https://github.com/chuanyangjin/fast-DiT
- Convolutions for Sequence Modeling: https://github.com/HazyResearch/safari
- Mamba SSM architecture: https://github.com/state-spaces/mamba
- Causal depthwise conv1d in CUDA, with a PyTorch interface: https://github.com/Dao-AILab/causal-conv1d
- Experiment Tracking with Weights and Biases : https://www.wandb.com/

## Citation

```bibtex
@article{chandrasegaran2025grafting,
      title={Exploring Diffusion Transformer Designs via Grafting},
      author={Chandrasegaran, Keshigeyan and Poli, Michael and Fu, Daniel Y. and Kim, Dongjun and 
      Hadzic, Lea M. and Li, Manling and Gupta, Agrim and Massaroli, Stefano and 
      Mirhoseini, Azalia and Niebles, Juan Carlos and Ermon, Stefano and Li, Fei-Fei},
      booktitle = {Advances in Neural Information Processing Systems},
      volume = {38}
      year={2025},
      url={https://arxiv.org/abs/2506.05340}, 
}
```
