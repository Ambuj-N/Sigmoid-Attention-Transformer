# Sigmoid-Attention-Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19uC9NqtvkabL798zN-dkXDNSd4IYZM0n)

A small PyTorch experiment inspired by Apple’s ICLR 2025 paper: *"[Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431)"*. 

The goal of this project is to check whether sigmoid-based attention can actually work as a replacement for softmax in practice. If you try "naive" sigmoid attention directly, training usually becomes unstable and the loss explodes early on. This repo implements a few key ideas from the paper that make it stable and usable.

### What’s inside

The implementation includes three main stabilizers:
* **Sequence Length Bias:** Adds a `-log(n)` term before the sigmoid to keep attention scores under control at the start.
* **QK Normalization:** Applies `LayerNorm` to queries and keys before computing similarity.
* **LayerScale:** Scales the attention output using a small learnable parameter initialized to `1e-4`.

There’s also a simple head-to-head comparison setup between:
1. **Sigmoid Attention** (with stabilizers)
2. **Standard Softmax Attention** (baseline)

Both models are trained on a toy sequence-copying task.

### Ablation Study: Sigmoid Attention Stabilizers

**Experimentation:** To understand why "naive" sigmoid attention fails, try removing the mathematical stabilizers in the `SigmoidAttention` class and watch how the training loss explodes:
* **LayerScale:** Change the initialization value from `1e-4` to a larger number like `1e-1`.
* **Sequence Bias:** Change the line `bias = -math.log(seq_length)` to `bias = 0`.
* **QK Normalization:** Comment out the `F.layer_norm` lines applied to the queries and keys. 

### Running the code

The setup is minimal. You only need PyTorch and Matplotlib.

```bash
git clone [https://github.com/Ambuj-N/Sigmoid-Attention-Transformer.git](https://github.com/Ambuj-N/Sigmoid-Attention-Transformer.git)
cd Sigmoid-Attention-Transformer
pip install -r requirements.txt
python train.py
