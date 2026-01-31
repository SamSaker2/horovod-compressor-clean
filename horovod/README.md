# horovod-compressor-final 
# Random Sparsity Compressor for Horovod (TensorFlow + PyTorch)

This repository contains an implementation of a **deterministic 30% Random Sparsification Compressor** for **Horovod**, supporting both **TensorFlow** and **PyTorch**.  
The implementation reduces gradient communication overhead by sending only a fixed percentage of entries (30%) while ensuring **identical sampling across all workers** without transmitting indices.

---

## 🚀 Features

### ✔ Deterministic Sampling Across Workers
Workers regenerate the **same random indices** using:
- **TensorFlow:** `tf.random.experimental.stateless_shuffle`  
- **PyTorch:** Seeded `torch.Generator`

This ensures synchronization with **zero communication overhead** for indices.

### ✔ TensorFlow Version (Graph-safe)
- No `.numpy()` calls  
- 100% compatible with TF1 Graph mode + TF2 Eager  
- Stores shapes and sizes dynamically in `ctx` without breaking graph execution

### ✔ PyTorch Version
- Stores original `dtype` in `ctx`  
- Uses `dtype` correctly during decompression  
- Zero-allocation-based reconstruction

### ✔ Horovod-Compatible
Drop-in support using:
```python
optimizer = hvd.DistributedOptimizer(optimizer, compression=RandomSparsityCompressor)
