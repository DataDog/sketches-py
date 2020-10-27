# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

import math

import numpy as np

from .store import CollapsingLowestDenseStore


DEFAULT_REL_ACC = 0.01  # "alpha" in the paper
DEFAULT_BIN_LIMIT = 2048
DEFAULT_MIN_VALUE = 1.0e-9


class UnequalSketchParametersException(Exception):
    pass


class BaseDDSketch:
    def __init__(
        self,
        store,
        negative_store,
        relative_accuracy=None,
        bin_limit=None,
    ):
        self.store = store
        self.negative_store = negative_store

        # Make sure the parameters are valid
        if relative_accuracy is None or (
            relative_accuracy <= 0 or relative_accuracy >= 1
        ):
            relative_accuracy = DEFAULT_REL_ACC

        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT

        self.zero_count = 0

        gamma_mantissa = 2 * relative_accuracy / (1 - relative_accuracy)
        self.gamma = 1 + gamma_mantissa
        gamma_ln = math.log1p(gamma_mantissa)
        self.multiplier = 1.0 / gamma_ln
        self.min_value = np.finfo(np.float64).tiny * self.gamma

        self.min = float("+inf")
        self.max = float("-inf")
        self.count = 0
        self._sum = 0.0

    def __repr__(self):
        return "store: {{{}}}, negative_store: {{{}}}, zero_count: {}, count: {}, sum: {}, min: {}, max: {}".format(
            self.store,
            self.negative_store,
            self.zero_count,
            self.count,
            self._sum,
            self.min,
            self.max,
        )

    @property
    def name(self):
        return "DDSketch"

    @property
    def num_values(self):
        return self.count

    @property
    def avg(self):
        return self._sum / self.count

    @property
    def sum(self):
        return self._sum

    def add(self, val):
        """Add a value to the sketch."""
        if val > self.min_value:
            key = int(math.ceil(math.log(val) * self.multiplier))
            self.store.add(key)
        elif val < -self.min_value:
            key = int(math.ceil(math.log(-val) * self.multiplier))
            self.negative_store.add(key)
        else:
            self.zero_count += 1

        # Keep track of summary stats
        self.count += 1
        self._sum += val
        if val < self.min:
            self.min = val
        if val > self.max:
            self.max = val

    def get_quantile_value(self, quantile):
        if quantile < 0 or quantile > 1 or self.count == 0:
            return np.NaN
        if quantile == 0:
            return self.min
        if quantile == 1:
            return self.max

        rank = int(quantile * (self.count - 1) + 1)

        if rank <= self.negative_store.count:
            key = self.negative_store.reversed_key_at_rank(rank)
            quantile_value = -2 * pow(self.gamma, key) / (1 + self.gamma)
        elif rank <= self.zero_count + self.negative_store.count:
            return 0
        else:
            key = self.store.key_at_rank(
                rank - self.zero_count - self.negative_store.count
            )
            quantile_value = 2 * pow(self.gamma, key) / (1 + self.gamma)

        return max(quantile_value, self.min)

    def merge(self, sketch):
        if not self.mergeable(sketch):
            raise UnequalSketchParametersException(
                "Cannot merge two DDSketches with different parameters"
            )

        if sketch.count == 0:
            return

        if self.count == 0:
            self.copy(sketch)
            return

        # Merge the stores
        self.store.merge(sketch.store)
        self.negative_store.merge(sketch.negative_store)
        self.zero_count += sketch.zero_count

        # Merge summary stats
        self.count += sketch.count
        self._sum += sketch.sum
        if sketch.min < self.min:
            self.min = sketch.min
        if sketch.max > self.max:
            self.max = sketch.max

    def mergeable(self, other):
        """Two sketches can be merged only if their gammas are equal."""
        return self.gamma == other.gamma

    def copy(self, sketch):
        self.store.copy(sketch.store)
        self.negative_store.copy(sketch.negative_store)
        self.zero_count = sketch.zero_count
        self.min = sketch.min
        self.max = sketch.max
        self.count = sketch.count
        self._sum = sketch.sum


class DDSketch(BaseDDSketch):
    """The default implementation of a memory-optimal instance of BaseDDSketch, with
    optimized memory usage, at the cost of lower ingestion speed, using a
    limited number of bins. When the maximum number of bins is reached, bins
    with lowest indices are collapsed, which causes the relative accuracy to be
    lost on lowest quantiles. For the default bin limit, collapsing is unlikely
    to occur unless the data is distributed with tails heavier than any
    subexponential. (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
    """

    def __init__(self, relative_accuracy=None, bin_limit=None):
        if bin_limit is None or bin_limit < 0:
            bin_limit = DEFAULT_BIN_LIMIT
        store = CollapsingLowestDenseStore(bin_limit)
        negative_store = CollapsingLowestDenseStore(bin_limit, initial_nbins=0)
        super().__init__(
            store=store,
            negative_store=negative_store,
            relative_accuracy=relative_accuracy,
            bin_limit=bin_limit,
        )
