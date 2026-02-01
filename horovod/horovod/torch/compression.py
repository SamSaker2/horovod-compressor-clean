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
#New: Deterministic Random 30% Sparsification
import hashlib
import torch

class RandomSparsityTorch(object):
    """
    Deterministic random sparsification compressor for PyTorch.

    The compressor randomly samples a constant number of elements 
    (default: 30%) of a tensor uniformly without replacement.
    The compressor uses a deterministic 64-bit seed that depends on the tensor’s dtype, shape, and a counter per signature.
    Thus, all workers will use the same indices without any communication.
    Only the sampled elements are sent. On decompression, 
    the same indices are generated locally with the same seed,
    and the tensor is reconstructed by filling a tensor of zeros with the original shape.
    """
    _counters = {}

    @staticmethod
    def _signature(tensor):
        """
        Builds a deterministic signature for the tensor based on its data type and size.
        Returns:
        A tuple (dtype_string, size_tuple) used to derive a stable seed.
        """
        return (str(tensor.dtype), tuple(tensor.size()))

    @staticmethod
    def _stable_base(sig):
        """
        From this signature, it generates a stable 31-bit integer seed using SHA-256.
        This ensures that the same seed will be generated across all workers if the layout of the tensor is the same.
        """
        h = hashlib.sha256(repr(sig).encode('utf-8')).digest()
        return int.from_bytes(h[:4], byteorder='big', signed=False) & 0x7FFFFFFF

    @classmethod
    def _next_seed(cls, sig):
        """
        Finally, it generates a deterministic 64-bit integer seed for the CPU RNG used by PyTorch.
        It combines this information from:
        1- A stable base, which is generated from the signature, and
        2- A counter, which is specific to the signature.
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
        This function compresses a tensor by choosing a deterministic random subset of its elements.
        How it works:
        1- Flatten the tensor. 2- Compute k as the ceiling of fraction times n.
        3- Initialize a cpu-based torch.Generator with a deterministic seed.
        4- Sample a random permutation. 5- Sample k indices out of the permutation and gather the tensor elements.
        Arguments:
        tensor: input PyTorch tensor.
        fraction: fraction of elements to keep (default: 0.30).
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
        The function rebuilds the original tensor by inserting the incoming values into a zero-padded buffer.
        Steps:
        1- Reconstruct the same random order based on the stored seed.
        2- Select the first k elements of the order.
        3- Create a flat buffer of size n with all elements as zero.
        4- Scatter the incoming values into the selected positions.
        5- Reshape the flat buffer back to the original tensor shape.
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
