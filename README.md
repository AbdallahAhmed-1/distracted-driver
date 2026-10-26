---
title: Distracted Driver Detection
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Distracted Driver Detection

An image classifier that looks at a photo of a driver and predicts what they are doing: safe driving, texting, talking on the phone, drinking, and so on. Built with fastai (ResNet34 transfer learning) and deployed as a Gradio app.

**Live demo:** https://huggingface.co/spaces/abdalla971/distracted_driver

## Classes

| Label | Behavior | Label | Behavior |
|---|---|---|---|
| c0 | Safe driving | c5 | Operating the radio |
| c1 | Texting (right hand) | c6 | Drinking |
| c2 | Phone (right hand) | c7 | Reaching behind |
| c3 | Texting (left hand) | c8 | Hair / makeup |
| c4 | Phone (left hand) | c9 | Talking to passenger |

## Dataset

[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (Kaggle): about 22,400 labeled training images of 26 drivers, taken from in-car video frames.

## Approach

**Validation split by driver, not by image.** Frames from the same driver are nearly identical, so a random split leaks near-duplicates into validation and gives an inflated score. I hold out 7 whole drivers, so the validation score reflects performance on people the model has never seen.

**Fighting overfitting.** My first run (5 held-out drivers, no regularization) memorized the training set: train loss fell to 0.006 while validation error bounced between 0.19 and 0.29. I fixed it with:

- **Augmentation:** small rotation, zoom, warp and lighting changes. Horizontal flip is disabled because it would swap left-hand and right-hand classes.
- **Label smoothing:** softer targets, so the model can't push its confidence to 100% by memorizing.
- **Weight decay (0.1):** discourages extreme, overly specific weights.
- **Best-epoch checkpointing:** keeps the epoch with the lowest validation loss.

**Training setup**

| Item | Value |
|---|---|
| Backbone | ResNet34 (ImageNet pretrained) |
| Input | 224x224 (center crop) |
| Batch size | 64, mixed precision (fp16) |
| Schedule | `fine_tune`, 1 frozen epoch + 4 unfrozen, base LR 1e-3 |
| Loss | Label-smoothing cross-entropy |

## Results

Validation error rate is about **0.20** on 7 held-out drivers.

| Epoch | Train loss | Valid loss | Error rate |
|---|---|---|---|
| 0 | 0.909 | 1.304 | 0.327 |
| 1 | 0.713 | 1.079 | 0.225 |
| 2 | 0.628 | 1.016 | 0.202 |
| 3 | 0.606 | 1.013 | 0.209 |

Loss values are higher than in an unregularized run because label smoothing raises the loss floor, so compare error rate, not loss.

## Deployment

The model is exported as **TorchScript** (`model.pt`) with class names in `vocab.json`, instead of a pickled fastai learner. A fastai pickle embeds Python bytecode and only loads on the exact Python version that created it, which broke loading on the Space. TorchScript needs only `torch`, so it runs on any Python version.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Gradio app: preprocessing and prediction |
| `model.pt` | TorchScript model (stored with Git LFS) |
| `vocab.json` | Class labels in model output order |
| `requirements.txt` | Dependencies (`torch`) |
| `notebook.ipynb` | Training notebook (Kaggle) |

## Run locally

```bash
git clone https://github.com/AbdallahAhmed-1/distracted-driver.git
cd distracted-driver
git lfs pull
pip install torch gradio pillow numpy
python app.py
```

## Limitations and next steps

- The model was trained on 26 drivers in a few cars, so accuracy can drop for unfamiliar cameras, angles or lighting.
- Validation covers only 7 drivers, so the score is an estimate, not a guarantee.
- Planned: larger input size (320), a stronger backbone, and retraining on all drivers for the final model.