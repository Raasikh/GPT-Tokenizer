# 🧠 GPT Tokenizer - Byte Pair Encoding from Scratch

Welcome to **GPT Tokenizer** — a simple and educational Byte Pair Encoding (BPE) tokenizer built entirely from scratch using PyTorch and Python. This tokenizer demonstrates how subword tokenization (as used in GPT models) works under the hood.

---

## 🚀 Features

- ✅ Byte-level tokenization (256 base tokens)
- ✅ Customizable vocabulary size
- ✅ BPE merge logic implemented from scratch
- ✅ Encoding and decoding support
- ✅ Fully reversible token-to-text pipeline
- ✅ Educational, readable code structure

---

## 🧩 What is Byte Pair Encoding (BPE)?

Byte Pair Encoding is a subword tokenization method used in GPT and other Transformer models. It works by:

1. Starting with individual bytes (0–255)
2. Iteratively merging the most frequent adjacent token pairs
3. Building a vocabulary of base bytes + learned subwords

This leads to compact token sequences and efficient learning.

---

## 🛠️ How It Works

### 1. Token Initialization
- Start with UTF-8 encoded bytes
- Each byte (0–255) is treated as a unique token

### 2. Merge Operations
- Count the most frequent adjacent token pairs
- Merge them to create new tokens
- Repeat until desired vocab size is reached

### 3. Encoding
```python
encoded = encode("hello world")
# -> [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
```

### 4. Decoding
```python
decoded = decode(encoded)
# -> "hello world"
```

---

## 📂 Repository Structure

```
.
├── tokenizer.py         # Core BPE logic: merge, encode, decode
├── input.txt            # Training corpus for building merge rules
├── README.md            # You're here!
```

---

## 🧪 Example Usage

```python
from tokenizer import encode, decode

text = "unbelievable machines"
encoded = encode(text)
decoded = decode(encoded)

print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)
```

---

## 📚 Learning Goals

This repo is built to help you:

- Understand BPE as used in GPT and LLM tokenizers
- Learn how vocab size impacts compression
- Explore how models tokenize and reconstruct text

---

## 📦 Requirements

- Python 3.7+
- PyTorch (optional, if used for tensor-based optimization)

---

## 🙌 Credits

Inspired by the teaching style of [Andrej Karpathy](https://github.com/karpathy)  
Created with ❤️ for learning and fun.

---
