# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

from abc import ABC, abstractmethod
import math

import numpy as np

"""
We start with 128 bins and grow the store in chunks of 128 unless specified otherwise.
"""
INITIAL_NBINS = 128
CHUNK_SIZE = 128


class Store(ABC):
    @abstractmethod
    def length(self, store):
        pass

    @abstractmethod
    def add(self, key):
        pass

    @abstractmethod
    def key_at_rank(self, rank):
        pass

    @abstractmethod
    def merge(self, store):
        pass

    @abstractmethod
    def copy(self, store):
        pass


class DenseStore(Store):
    def __init__(self, initial_nbins=INITIAL_NBINS, chunk_size=CHUNK_SIZE):
        self.initial_nbins = initial_nbins
        self.chunk_size = chunk_size
        self.initial_chunk_size = chunk_size

        self.count = 0
        self.min_key = 0
        self.max_key = 0
        self.bins = [0] * self.initial_nbins

    def __repr__(self):
        repr_str = "{"
        for i, sbin in enumerate(self.bins):
            repr_str += "{}: {}, ".format(i + self.min_key, sbin)
        repr_str += "}}, minKey: {}, maxKey: {}".format(self.min_key, self.max_key)
        return repr_str

    def length(self):
        return len(self.bins)

    def _grow_by(self, required_growth):
        return self.chunk_size * math.ceil((required_growth) / self.chunk_size)

    def add(self, key):
        if len(self.bins) == 0:
            self.bins = [0] * self.initial_chunk_size
        if self.count == 0:
            self.max_key = key
            self.min_key = key - len(self.bins) + 1
        elif key < self.min_key:
            self._grow_left(key)
        elif key > self.max_key:
            self._grow_right(key)

        idx = max(0, key - self.min_key)
        self.bins[idx] += 1
        self.count += 1

    def key_at_rank(self, rank):
        """Return the key for the value at given rank"""
        n = 0
        for i, bin_ct in enumerate(self.bins):
            n += bin_ct
            if n >= rank:
                return i + self.min_key
        return self.max_key

    def reversed_key_at_rank(self, rank):
        """Return the key for the value at given rank in reversed order"""
        n = 0
        for i, b in reversed(list(enumerate(self.bins))):
            n += b
            if n >= rank:
                return i + self.min_key
        return self.min_key

    def _grow_left(self, key):
        if self.min_key < key:
            return

        min_key = self.min_key - self._grow_by(self.min_key - key)

        self.bins[:0] = [0] * (self.min_key - min_key)
        self.min_key = min_key

    def _grow_right(self, key):
        if self.max_key > key:
            return

        max_key = self.max_key + self._grow_by(key - self.max_key)
        self.bins.extend([0] * (max_key - self.max_key))
        self.max_key = max_key

    def merge(self, store):
        if store.count == 0:
            return

        if self.count == 0:
            self.copy(store)
            return

        if self.max_key > store.max_key:
            if store.min_key < self.min_key:
                self._grow_left(store.min_key)

            for i in range(max(self.min_key, store.min_key), store.max_key + 1):
                self.bins[i - self.min_key] += store.bins[i - store.min_key]

            if self.min_key > store.min_key:
                n = np.sum(store.bins[: self.min_key - store.min_key])
                self.bins[0] += n
        else:
            if store.min_key < self.min_key:
                tmp = store.bins[:]
                for i in range(self.min_key, self.max_key + 1):
                    tmp[i - store.min_key] += self.bins[i - self.min_key]
                self.bins = tmp
                self.max_key = store.max_key
                self.min_key = store.min_key
            else:
                self._grow_right(store.max_key)
                for i in range(store.min_key, store.max_key + 1):
                    self.bins[i - self.min_key] += store.bins[i - store.min_key]

        self.count += store.count

    def copy(self, store):
        self.bins = store.bins[:]
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key


class CollapsingLowestDenseStore(DenseStore):
    def __init__(self, max_bins, initial_nbins=INITIAL_NBINS, chunk_size=CHUNK_SIZE):
        self.max_bins = max_bins
        self.initial_nbins = min(max_bins, initial_nbins)
        self.chunk_size = chunk_size
        self.initial_chunk_size = min(max_bins, chunk_size)

        self.count = 0
        self.min_key = 0
        self.max_key = 0

        self.bins = [0] * self.initial_nbins

    def _grow_left(self, key):
        if self.min_key < key or len(self.bins) >= self.max_bins:
            return

        min_possible = self.max_key - self.max_bins + 1
        if self.max_key - key >= self.max_bins:
            min_key = min_possible
        else:
            min_key = max(
                self.min_key - self._grow_by(self.min_key - key), min_possible
            )

        self.bins[:0] = [0] * (self.min_key - min_key)
        self.min_key = min_key

    def _grow_right(self, key):
        if self.max_key > key:
            return

        if key - self.max_key >= self.max_bins:
            # the new key is over max_bins to the right; put everything in the first bin
            self.bins = [0] * self.max_bins
            self.max_key = key
            self.min_key = key - self.max_bins + 1
            self.bins[0] = self.count
        elif key - self.min_key >= self.max_bins:
            # the new key requires us to compress on the left
            min_key = key - self.max_bins + 1
            n = np.sum(self.bins[: min_key - self.min_key])
            self.bins = self.bins[min_key - self.min_key :]
            self.bins.extend([0] * (key - self.max_key))
            self.max_key = key
            self.min_key = min_key
            self.bins[0] += n
        else:
            # grow to the right
            max_key = min(
                self.max_key + self._grow_by(key - self.max_key),
                self.min_key + self.max_bins,
            )
            self.bins.extend([0] * (max_key - self.max_key))
            self.max_key = max_key

    def copy(self, store):
        self.bins = store.bins[:]
        self.max_bins = store.max_bins
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key
