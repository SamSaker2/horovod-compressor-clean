# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
# === New: Deterministic Random 30% Sparsification (PyTorch) ===
import hashlib
import torch

class RandomSparsityTorch(object):
    """
    Deterministic random sparsification compressor for PyTorch.

    This compressor selects a fixed fraction (default 30%) of tensor elements
    using uniform random sampling without replacement. A deterministic 64-bit
    seed is derived from the tensor's dtype, shape, and a per-signature counter,
    ensuring that all workers generate identical random indices without any
    communication.

    Only the selected values are transmitted. During decompression, the same
    indices are regenerated locally using the same seed, and the values are
    scattered back into a zero-filled tensor of the original shape.
    """
    _counters = {}

    @staticmethod
    def _signature(tensor):
        """
        Builds a deterministic signature for the tensor based on dtype and size.

        Returns:
            A tuple (dtype_string, size_tuple) used to derive a stable seed.
        """
        return (str(tensor.dtype), tuple(tensor.size()))

    @staticmethod
    def _stable_base(sig):
        """
        Computes a stable 31-bit integer seed from the tensor signature using SHA-256.

        Ensures identical seeds across all workers for the same tensor structure.
        """
        h = hashlib.sha256(repr(sig).encode('utf-8')).digest()
        return int.from_bytes(h[:4], byteorder='big', signed=False) & 0x7FFFFFFF

    @classmethod
    def _next_seed(cls, sig):
        """
        Generates a deterministic 64-bit seed for PyTorch's CPU RNG.

        Combines:
            - a stable base derived from the tensor signature
            - a per-signature counter

        Returns:
            (seed64, step): a 64-bit integer seed and the counter value.
        """
        step = cls._counters.get(sig, 0)
        cls._counters[sig] = step + 1
        base = cls._stable_base(sig)
        seed64 = ((base << 1) ^ (step & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        return seed64, step

    @classmethod
    def compress(cls, tensor, fraction=0.30):
        """
        Compresses the tensor by selecting a deterministic random subset of values.

        Steps:
            1. Flatten the tensor.
            2. Compute k = ceil(fraction * n).
            3. Create a CPU-based torch.Generator with the deterministic seed.
            4. Generate a random permutation using torch.randperm.
            5. Select the first k indices and gather the corresponding values.

        Args:
            tensor: Input PyTorch tensor.
            fraction: Fraction of elements to keep (default 0.30).

        Returns:
            values: 1-D tensor of selected values.
            ctx: A tuple containing (original_size, seed64, n, k, device, dtype).
        """
        if not tensor.is_floating_point():
            return tensor, None
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1].")

        sig = cls._signature(tensor)
        seed64, _ = cls._next_seed(sig)

        flat = tensor.contiguous().view(-1)
        n = flat.numel()
        k = max(1, int(fraction * n + 0.999999))

        g = torch.Generator(device='cpu')
        g.manual_seed(seed64)
        idx_cpu = torch.randperm(n, generator=g, device='cpu')[:k].to(dtype=torch.int64)

        idx = idx_cpu.to(device=tensor.device)
        values = flat.index_select(0, idx)

        ctx = (tuple(tensor.size()), seed64, n, k, tensor.device, tensor.dtype)
        return values, ctx

    @classmethod
    def decompress(cls, values, ctx):
        """
        Reconstructs the original tensor by scattering values into a zero-filled buffer.

        Steps:
            1. Regenerate the same random permutation using the stored seed.
            2. Select the first k indices.
            3. Create a zero-initialized flat tensor of length n.
            4. Scatter the received values at the selected indices.
            5. Reshape back to the original tensor size.

        Args:
            values: 1-D tensor of received values.
            ctx: (original_size, seed64, n, k, device, dtype).

        Returns:
            A PyTorch tensor with the original shape and zeros in unselected positions.
        """
        if ctx is None:
            return values
        original_size, seed64, n, k, device, dtype = ctx

        g = torch.Generator(device='cpu')
        g.manual_seed(seed64)
        idx_cpu = torch.randperm(n, generator=g, device='cpu')[:k].to(dtype=torch.int64)
        idx = idx_cpu.to(device=device)

        flat = torch.zeros(n, device=device, dtype=dtype)
        flat.index_copy_(0, idx, values.to(dtype))
        return flat.view(original_size)

class RandomSparsityCompressor(RandomSparsityTorch):
    pass
