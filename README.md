# CycleGAN-With-Attention-Using-PyTorch

This repository contains three main implementations in PyTorch of CycleGAN-based models for unpaired image-to-image translation, inspired by the original [CycleGAN paper](https://arxiv.org/abs/1703.10593).  
The project compares standard CycleGAN, an attention-augmented CycleGAN, and an ablation study version without augmentations.

---

## 📂 Repository Structure

- **CycleGAN/**  
  Standard implementation of CycleGAN.  
  - Shares the same `train.py` with `CycleGAN-Attention`, but has a different `models.py`.

- **CycleGAN-Attention/**  
  CycleGAN enhanced with an attention mechanism.  
  - Same `train.py` as `CycleGAN`.  
  - Modified `models_attention.py` to integrate attention.

- **CycleGAN-Attention_Without_augmentations/**  
  Ablation study to evaluate the effect of data augmentation.  
  - Shares the same `models_attention.py` as `CycleGAN-Attention`.  
  - Different `train.py` (without augmentations).

Each folder contains the following files:
- `MIFID.py` – Implementation of Memorization-Informed Fréchet Inception Distance (evaluation metric).  
- `datasets.py` – Dataset loading and preprocessing utilities.  
- `models.py` or `models_attention.py` – Model definitions (without/with attention, respectively).  
- `split_data.py` – Script to prepare and split dataset.  
- `test.py` – Testing script for generating translated images.  
- `train.py` – Training loop and experiment setup.  
- `utils.py` – Helper functions.

---

## 📖 Reference

- Original CycleGAN Paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  
- Official GitHub implementation: [aitorzip/PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN/tree/master?tab=readme-ov-file)

---

## 🚀 Running Instructions

### 1. Dataset
Download the Monet dataset from Kaggle:  
[GAN Getting Started - Monet2Photo](https://www.kaggle.com/competitions/gan-getting-started/data)

### 2. Prepare Data
From any model folder:
```bash
python split_data.py --src
```

---

### 🔹 Model 1: CycleGAN
```bash
cd CycleGAN

# Train
python ./train --dataroot "./datasets/monet2photo" --cuda --n_cpu 0

# Start Visdom (in another terminal)
python -m visdom.server
# Monitor training in browser: http://localhost:8097/

# Test
python ./test.py   --dataroot "./datasets/monet2photo"   --size 256   --batchSize 1   --n_cpu 0   --cuda   --generator_A2B ./output/netG_A2B.pth   --generator_B2A ./output/netG_B2A.pth   --mifid AtoB   --mifid_max 500   --tau 0.2
```

#### 🖼️ Example Results after 40 Epochs (Left - before, Right - after):
<img width="518" height="260" alt="0034" src="https://github.com/user-attachments/assets/11033a0e-effd-4859-b30f-61a546ccc346" />
<img width="518" height="260" alt="0035" src="https://github.com/user-attachments/assets/b7c2ec5d-b1e2-4b40-aec3-1371947fd315" />

---

### 🔹 Model 2: CycleGAN-Attention
```bash
cd CycleGAN-Attention

# Train
python ./train --dataroot "./datasets/monet2photo" --cuda --n_cpu 0

# Start Visdom (in another terminal)
python -m visdom.server
# Monitor training in browser: http://localhost:8097/

# Test
python ./test.py   --dataroot "./datasets/monet2photo"   --size 256   --batchSize 1   --n_cpu 0   --cuda   --generator_A2B ./output/netG_A2B.pth   --generator_B2A ./output/netG_B2A.pth   --mifid AtoB   --mifid_max 500   --tau 0.2
```

#### 🖼️ Example Results after 40 Epochs (Left - before, Right - after):
<img width="518" height="260" alt="0034" src="https://github.com/user-attachments/assets/4e398e5f-578f-4070-826c-d95ccb1b6512" />
<img width="518" height="260" alt="0035" src="https://github.com/user-attachments/assets/2d476dc1-75be-4c4c-92e0-e41869c906a8" />

---

### 🔹 Ablation Study: CycleGAN-Attention_Without_augmentations
```bash
cd CycleGAN-Attention_Without_augmentations

# Train
python ./train --dataroot "./datasets/monet2photo" --cuda --n_cpu 0

# Start Visdom (in another terminal)
python -m visdom.server
# Monitor training in browser: http://localhost:8097/

# Test
python ./test.py   --dataroot "./datasets/monet2photo"   --size 256   --batchSize 1   --n_cpu 0   --cuda   --generator_A2B ./output/netG_A2B.pth   --generator_B2A ./output/netG_B2A.pth   --mifid AtoB   --mifid_max 500   --tau 0.2
```

---

## 📊 Experiments

- **Model 1 – CycleGAN**  
  Baseline implementation.

- **Model 2 – CycleGAN-Attention**  
  Introduces attention mechanisms into the generator/discriminator.

- **Ablation – CycleGAN-Attention_Without_augmentations**  
  Tests the impact of removing augmentations, while keeping the attention architecture.

---

## 📜 License
See the [LICENSE](./LICENSE) file.
