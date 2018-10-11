# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2018 Datadog, Inc.

import math

import numpy as np

from store import Store


DEFAULT_ALPHA = 0.01
DEFAULT_BIN_LIMIT = 2048
DEFAULT_MIN_VALUE = 1.0e-9
        

class UnequalSketchParametersException(Exception):
    pass


class DogSketch(object):

    def __init__(self, alpha=None, bin_limit=None, min_value=None):
        # Make sure the parameters are valid
        if alpha is None or (alpha <= 0 or alpha >= 1):
            alpha = DEFAULT_ALPHA
        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT
        if min_value < 0:
            min_value = DEFAULT_MIN_VALUE

        self.gamma = 1 + 2*alpha
        self.gamma_ln = math.log1p(2*alpha)
        self.min_value = min_value
        self.offset = -int(math.ceil(math.log(min_value)/self.gamma_ln)) + 1

        self.store = Store(bin_limit)
        self._min = float('+inf') 
        self._max = float('-inf') 
        self._count = 0
        self._sum = 0

    def __repr__(self):
        return "store: {{{}}}, count: {}, sum: {}, min: {}, max: {}".format(
            self.store, self.count, self._sum, self.min, self.max)

    @property
    def name(self):
        return 'DogSketch'

    @property
    def num_values(self):
        return self._count

    @property
    def avg(self):
        return float(self._sum)/self._count

    @property
    def sum(self):
        return self._sum

    def get_key(self, val):
        if val < -self.min_value:
            return -int(math.ceil(math.log(-val)/self.gamma_ln)) - self.offset
        elif val > self.min_value:
            return int(math.ceil(math.log(val)/self.gamma_ln)) + self.offset
        else:
            return 0

    def add(self, val):
        """ Add a value to the sketch.
        """
        key = self.get_key(val)
        self.store.add(key)

        # Keep track of summary stats
        self._count += 1
        self._sum += val
        if val < self._min:
            self._min = val
        if val > self._max:
            self._max = val

    def quantile(self, q):
        if q < 0 or q > 1 or self._count == 0:
            return np.NaN
        if q == 0:
            return self._min
        if q == 1:
            return self._max

        rank = int(q*(self._count - 1) + 1)
        key = self.store.key_at_rank(rank)
        if key < 0:
                    key += self.offset
                    quantile = -0.5*(1 + self.gamma)*pow(self.gamma, -key-1)
        elif key > 0:
                    key -= self.offset
                    quantile = 0.5*(1 + self.gamma)*pow(self.gamma, key-1)
        else:
            quantile = 0

        return  max(quantile, self._min)
            
    def merge(self, sketch):
        if not self.mergeable(sketch):
            raise UnequalSketchParametersException("Cannot merge two DogSketches with different parameters")

        if sketch._count == 0:
            return

        if self._count == 0:
            self.copy(sketch)
            return

        # Merge the stores
        self.store.merge(sketch.store)

        # Merge summary stats
        self._count += sketch._count
        self._sum += sketch._sum
        if sketch._min < self._min:
            self._min = sketch._min
        if sketch._max > self._max:
            self._max = sketch._max

    def mergeable(self, other):
        """ Two sketches can be merged only if their gamma and min_values are equal.
        """
        return self.gamma == other.gamma and self.min_value == other.min_value

    def copy(self, sketch):
        self.store.copy(sketch.store) 
        self._min = sketch._min
        self._max = sketch._max
        self._count = sketch._count
        self._sum = sketch._sum
