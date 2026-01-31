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

import tensorflow as tf


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
        if tensor.dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
# === New: Deterministic Random 30% Sparsification (graph-safe) ===
import hashlib
import tensorflow as tf

class RandomSparsityTF(object):
    """Random p-sparsification with deterministic shared randomness (TensorFlow)."""
    _counters = {}

    @staticmethod
    def _static_shape_tuple(t):
        if hasattr(t, "shape") and t.shape.rank is not None:
            return tuple(d if d is None else int(d) for d in t.shape.as_list())
        return tuple()

    @classmethod
    def _signature(cls, tensor):
        shp = cls._static_shape_tuple(tensor)
        return (str(tensor.dtype.name), shp)

    @staticmethod
    def _stable_base(sig):
        h = hashlib.sha256(repr(sig).encode('utf-8')).digest()
        return int.from_bytes(h[:4], byteorder='big', signed=False) & 0x7FFFFFFF

    @classmethod
    def _next_seed(cls, sig):
        step = cls._counters.get(sig, 0)
        cls._counters[sig] = step + 1
        base = cls._stable_base(sig)
        return tf.constant([base, step & 0x7FFFFFFF], dtype=tf.int32)

    @classmethod
    def compress(cls, tensor, fraction=0.30):
        if not tensor.dtype.is_floating:
            return tensor, None
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1].")

        sig = cls._signature(tensor)
        seed = cls._next_seed(sig)

        flat = tf.reshape(tensor, [-1])
        n = tf.size(flat)
        k = tf.maximum(1, tf.cast(tf.math.ceil(tf.cast(n, tf.float32) * fraction), tf.int32))

        try:
            perm = tf.random.experimental.stateless_shuffle(tf.range(n, dtype=tf.int32), seed=seed)
        except AttributeError:
            perm = tf.random.stateless_shuffle(tf.range(n, dtype=tf.int32), seed=seed)
        idx = perm[:k]
        values = tf.gather(flat, idx)

        ctx = (tf.shape(tensor), seed, n)
        return values, ctx

    @classmethod
    def decompress(cls, values, ctx):
        if ctx is None:
            return values
        original_shape, seed, n = ctx
        k = tf.size(values)

        try:
            perm = tf.random.experimental.stateless_shuffle(tf.range(n, dtype=tf.int32), seed=seed)
        except AttributeError:
            perm = tf.random.stateless_shuffle(tf.range(n, dtype=tf.int32), seed=seed)
        idx = perm[:k]

        flat = tf.zeros([n], dtype=values.dtype)
        flat = tf.tensor_scatter_nd_update(flat, tf.expand_dims(idx, 1), values)
        return tf.reshape(flat, original_shape)

class RandomSparsityCompressor(RandomSparsityTF):
    pass