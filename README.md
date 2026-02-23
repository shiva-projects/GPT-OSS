# GPT-OSS: Mixture-of-Experts Transformer From Scratch

## Overview

GPT-OSS is a from-scratch implementation of a decoder-only Transformer language model with a Mixture-of-Experts (MoE) feed-forward block.

The project focuses on implementing modern LLM architectural components manually, without relying on high-level transformer libraries. It is designed to provide a deep understanding of how contemporary GPT-style models are structured internally.

This implementation includes:

* Pre-Norm Transformer blocks
* RMSNorm instead of LayerNorm
* Rotary Positional Embeddings (RoPE)
* Grouped Query Attention (GQA) with sliding window support
* Causal masking
* Mixture-of-Experts feed-forward layer with top-k routing
* SwiGLU expert activation
* Weight tying between embedding and LM head




---

## Architecture

The model follows a decoder-only transformer design.

High-level flow:
<img width="4230" height="3847" alt="image" src="https://github.com/user-attachments/assets/68288d44-f92f-45c4-9e83-f2cccbb796c0" />


### Attention Block (PreNorm)

Each attention block consists of:

* RMSNorm
* QKV Linear projection
* Rotary Positional Embeddings (RoPE)
* Grouped Query Attention (GQA)
* Sliding window attention
* Causal + window masking
* Softmax
* Attention dropout
* Output projection
* Residual connection with dropout

Key characteristics:

* Pre-normalization for training stability
* Efficient attention via GQA
* Long-context handling via sliding window
* Strict autoregressive causal masking

---

### Mixture-of-Experts (MoE) Block (PreNorm)

Instead of a standard MLP feed-forward layer, this model uses a Mixture-of-Experts architecture.

Structure:

* RMSNorm
* Router gate (top-k selection)
* Multiple experts (SwiGLU feed-forward networks)
* Weighted combination of selected experts
* Residual dropout
* Residual connection

Each expert:

* Linear → SwiGLU → Linear

Routing:

* Tokens are dynamically routed to top-k experts
* Outputs are combined via weighted sum

This increases parameter capacity without proportional increase in per-token compute.

---

### Weight Tying

The embedding matrix and output projection (LM Head) share weights.

This:

* Reduces parameter count
* Improves training stability
* Aligns input and output token spaces

---

## Technical Components

* Framework: PyTorch
* Normalization: RMSNorm
* Activation: SwiGLU
* Positional Encoding: RoPE
* Attention: GQA + sliding window
* Feed-forward: Mixture-of-Experts
* Optimization: Adam/AdamW
* Loss: Cross-Entropy
* Training: Autoregressive next-token prediction

---

## Project Structure

```
GPT-OSS/
│
├── main.ipynb                 # Full implementation and training pipeline
├── checkpoints_strong/        # Saved model checkpoints
├── cache/                     # Cached intermediate files
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/shiva-projects/GPT-OSS.git
cd GPT-OSS
```

### 2. Install dependencies

```
pip install torch numpy tqdm matplotlib
```

### 3. Run training

Open:

```
jupyter notebook
```

Run `main.ipynb`.
