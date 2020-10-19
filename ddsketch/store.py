# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

from abc import ABC, abstractmethod

import numpy as np


INITIAL_NBINS = 128


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
    def __init__(self):
        self.count = 0
        self.min_key = 0
        self.max_key = 0
        self.bins = [0] * INITIAL_NBINS

    def __repr__(self):
        repr_str = "{"
        for i, b in enumerate(self.bins):
            repr_str += "{}: {}, ".format(i + self.min_key, b)
        repr_str += "}}, minKey: {}, maxKey: {}".format(self.min_key, self.max_key)
        return repr_str

    def length(self):
        return len(self.bins)

    def add(self, key):
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
        for i, b in enumerate(self.bins):
            n += b
            if n >= rank:
                return i + self.min_key
        return self.max_key

    def _grow_left(self, key):
        if self.min_key < key:
            return

        self.bins[:0] = [0] * (self.min_key - key)
        self.min_key = key

    def _grow_right(self, key):
        if self.max_key > key:
            return

        self.bins.extend([0] * (key - self.max_key))
        self.max_key = key

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
    def __init__(self, max_bins):
        self.max_bins = max_bins
        self.count = 0
        self.min_key = 0
        self.max_key = 0
        # self.is_collapsed = False
        self.bins = [0] * min(max_bins, INITIAL_NBINS)

    def _grow_left(self, key):
        if self.min_key < key or len(self.bins) >= self.max_bins:
            return

        if self.max_key - key >= self.max_bins:
            min_key = self.max_key - self.max_bins + 1
        else:
            min_key = self.min_key
            while min_key > key:
                min_key -= self.max_bins - (self.max_key - self.min_key) - 1

        self.bins[:0] = [0] * (self.min_key - min_key)
        self.min_key = min_key

    def _grow_right(self, key):
        if self.max_key > key:
            return

        if key - self.max_key >= self.max_bins:
            self.bins = [0] * self.max_bins
            self.max_key = key
            self.min_key = key - self.max_bins + 1
            self.bins[0] = self.count
        elif key - self.min_key >= self.max_bins:
            min_key = key - self.max_bins + 1
            n = np.sum(self.bins[: min_key - self.min_key])
            self.bins = self.bins[min_key - self.min_key :]
            self.bins.extend([0] * (key - self.max_key))
            self.max_key = key
            self.min_key = min_key
            self.bins[0] += n
        else:
            self.bins.extend([0] * (key - self.max_key))
            self.max_key = key

    def copy(self, store):
        self.bins = store.bins[:]
        self.max_bins = store.max_bins
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key
