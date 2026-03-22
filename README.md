# Sigmoid-Attention-Transformer

A small PyTorch experiment inspired by Apple’s ICLR 2025 paper *"Theory, Analysis, and Best Practices for Sigmoid Self-Attention"*.

The goal of this project is: check whether sigmoid-based attention can actually work as a replacement for softmax in practice.

If you try sigmoid attention directly, training usually becomes unstable and the loss explodes early on. This repo implements a few key ideas from the paper that make it stable and usable.

## What’s inside

The implementation includes three main stabilizers:

* **Sequence Length Bias**
  Adds a `-log(n)` term before the sigmoid to keep attention scores under control at the start.

* **QK Normalization**
  Applies LayerNorm to queries and keys before computing similarity.

* **LayerScale**
  Scales the attention output using a small learnable parameter initialized to `1e-4`.

There’s also a simple comparison setup between:

* Sigmoid Attention (with stabilizers)
* Standard Softmax Attention (baseline)

Both models are trained on a toy copy task.

## Running the code

The setup is minimal. You only need PyTorch and Matplotlib.

```bash
git clone https://github.com/Ambuj-N/Sigmoid-Attention-Transformer.git
cd Sigmoid-Attention-Transformer
pip install -r requirements.txt
python train.py
```

## Output

After training, the script generates a plot showing the loss curves of both models:

* `loss_comparison.png`

This gives a quick visual comparison of training stability and convergence.
