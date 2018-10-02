# Unless explicitly stated otherwise all files in this repository are licensed
# under the BSD-3-Clause License.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2018 Datadog, Inc.

import numpy as np


class Store(object):
   
    def __init__(self, max_bins):
        self.max_bins = max_bins
        self.bins = [0]*max(1, max_bins/16)
        self.count = 0
        self.min_key = 0
        self.max_key = 0

    def __repr__(self):
        repr_str = "{"
        for i, b in enumerate(self.bins):
            repr_str += "{}: {}, ".format(i+self.min_key, b)
        repr_str += "}}, minKey: {}, maxKey: {}".format(self.min_key, self.max_key)
        return repr_str

    def length(self):
        return len(self.bins)

    def add(self, key):
        if self.count == 0:
            self.max_key = key
            self.min_key = key - len(self.bins) + 1
        elif key < self.min_key:
            self.grow_left(key)
        elif key > self.max_key:
            self.grow_right(key)

        idx = max(0, key - self.min_key)
        self.bins[idx] += 1
        self.count += 1

    def grow_left(self, key):
        if self.min_key < key or len(self.bins) >= self.max_bins:
            return

        if self.max_key - key >= self.max_bins:
            nbins = self.max_bins - len(self.bins)
            self.bins[:0] = [0]*nbins
            self.min_key -= nbins
        else:
            self.bins[:0] = [0]*(self.min_key - key)
            self.min_key = key

    def grow_right(self, key):
        if self.max_key > key:
            return

        if key - self.max_key >= self.max_bins:
            self.bins = [0]*self.max_bins
            self.max_key = key
            self.min_key = key - self.max_bins + 1
            self.bins[0] = self.count
        elif key - self.min_key >= self.max_bins:
            self.bins.extend([0]*(key - self.max_key))
            self.max_key = key
            self.compress()
        else:
            self.bins.extend([0]*(key - self.max_key))
            self.max_key = key

    def compress(self):
        if len(self.bins) <= self.max_bins:
            return

        n = np.sum(self.bins[:len(self.bins) - self.max_bins])
        self.bins = self.bins[len(self.bins) - self.max_bins:]
        self.bins[0] += n
        self.min_key = self.max_key - self.max_bins + 1

    def merge(self, store):
        if store.count == 0:
            return
        
        if self.count == 0:
            self.bins = store.bins[:]
            self.count = store.count
            self.min_key = store.min_key
            self.max_key = store.max_key
            
        # extend bins to the right
        if store.max_key > self.max_key:
            self.bins.extend([0]*(store.max_key - self.max_key))
            self.max_key = store.max_key
        # extend bins to the left
        if len(self.bins) <= self.max_bins and store.min_key < self.min_key:
            nbins = min(self.max_bins - len(self.bins), self.min_key - store.min_key) 
            self.bins[:0] = [0]*nbins
            self.min_key = self.max_key - len(self.bins) + 1

        for k, c in enumerate(store.bins):
            idx = max(0, k + store.min_key - self.min_key)
            self.bins[idx] += c
        self.count += store.count

    def copy(self, store):
        self.bins = store.bins[:]
        self.max_bins = store.max_bins
        self.count = store.count
        self.min_key = store.min_key
        self.max_key = store.max_key
