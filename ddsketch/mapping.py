# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""A mapping between values and integer indices that imposes relative accuracy
guarantees. Specifically, for any value minIndexableValue() < value < maxIndexableValue
()}, implementations of {@link IndexMapping} must be such that {@code value(index(v))}
is close to {@code v} with a relative error that is less than {@link #relativeAccuracy()}.

In implementations of {@code IndexMapping}, there generally is a trade-off
between the cost of computing index and the number of indices that are required
to cover a given range of values (memory optimality). The mos* memory-optimal
mapping is the {@link LogarithmicMapping}, but it requires the costly evaluation
of the arithm when computing the index. Other mappings can approximate the
logarithmic mapping, while being less computonally costly. The following table
shows the characteristics of a few implementations of {@code IndexMapping},
highlighting the above-mentioned trade-off.
"""

from abc import ABC, abstractmethod
import math

import numpy as np


class KeyMapping(ABC):
    """
    Args:
        relative_accuracy (float): the accuracy guarantee; referred to as alpha
            in the paper. (0. < alpha < 1.)

    Attributes:
        gamma (float): the base for the exponential buckets
            gamma = (1 + alpha) / (1 - alpha)
        multiplier (float): used for calculating log_gamma(value)
            multiplier = 1 / log(gamma)
        min_possible: the smallest value the sketch can distinguish from 0
        max_possible: the largest value the sketch can handle
    """

    def __init__(self, relative_accuracy):
        self.relative_accuracy = relative_accuracy
        gamma_mantissa = 2 * relative_accuracy / (1 - relative_accuracy)
        self.gamma = 1 + gamma_mantissa

    @abstractmethod
    def key(self, value):
        """
        Args:
            value (float)
        Returns:
            int: the key specifying the bucket for value
        """

    @abstractmethod
    def value(self, key):
        """
        Args:
            key (int)
        Returns:
            float: the value represented by the bucket specified by the key
        """


class LogarithmicMapping(KeyMapping):
    """A memory-optimal KeyMapping, i.e., given a targeted relative accuracy, it
    requires the least number of keys to cover a given range of values. This is
    done by logarithmically mapping floating-point values to integers.
    """

    def __init__(self, relative_accuracy):
        super().__init__(relative_accuracy)
        self.multiplier = 1.0 / math.log(self.gamma)
        self.min_possible = np.finfo(np.float64).tiny * self.gamma
        self.max_possible = np.finfo(np.float64).max / self.gamma

    def key(self, value):
        return int(math.ceil(math.log(value) * self.multiplier))

    def value(self, key):
        return pow(self.gamma, key) * (2.0 / (1 + self.gamma))


class LinearlyInterpolatedMapping(KeyMapping):
    """A fast KeyMapping that approximates the memory-optimal one
    (LogarithmicMapping) by extracting the floor value of the logarithm to the
    base 2 from the binary representations of floating-point values and
    linearly interpolating the logarithm in-between."""

    def __init__(self, relative_accuracy):
        super().__init__(relative_accuracy)
        self.multiplier = 1.0 / math.log(self.gamma)
        self.min_possible = np.finfo(np.float64).tiny * self.gamma
        self.max_possible = np.finfo(np.float64).max / self.gamma
        self._log_correction = 1.0 / math.log2(math.e)

    def key(self, value):
        return int(math.ceil(math.log2(value) * self.multiplier * self._log_correction))

    def value(self, key):
        return pow(self.gamma, key) * (2.0 / (1 + self.gamma))
