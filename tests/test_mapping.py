# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""Tests for the KeyMapping classes"""

import abc
import math
from unittest import TestCase

import numpy
import pytest
import six

from ddsketch.mapping import CubicallyInterpolatedMapping
from ddsketch.mapping import LinearlyInterpolatedMapping
from ddsketch.mapping import LogarithmicMapping
from ddsketch.mapping import _cbrt


def _relative_error(expected_min, expected_max, actual):
    """Calculate the relative error"""
    if expected_min < 0 or expected_max < 0 or actual < 0:
        raise Exception()
    if (expected_min <= actual) and (actual <= expected_max):
        return 0.0
    if expected_min == 0 and expected_max == 0:
        return 0.0 if actual == 0 else float("+inf")
    if actual < expected_min:
        return (expected_min - actual) / expected_min

    return (actual - expected_max) / expected_max


def _test_value_rel_acc(mapping):
    """Calculate the relative accuracy of a mapping on a large range of values"""
    value_mult = 2 - math.sqrt(2) * 1e-1
    max_relative_acc = 0.0
    value = mapping.min_possible
    while value < mapping.max_possible / value_mult:
        value *= value_mult
        map_val = mapping.value(mapping.key(value))
        rel_err = _relative_error(value, value, map_val)
        assert rel_err < mapping.relative_accuracy
        max_relative_acc = max(max_relative_acc, rel_err)
    max_relative_acc = max(
        max_relative_acc,
        _relative_error(
            mapping.max_possible,
            mapping.max_possible,
            mapping.value(mapping.key(mapping.max_possible)),
        ),
    )
    return max_relative_acc


class BaseTestKeyMapping(six.with_metaclass(abc.ABCMeta)):
    """Abstract class for testing KeyMapping classes"""

    offsets = [0, 1, -12.23, 7768.3]

    @abc.abstractmethod
    def mapping(self, relative_accuracy, offset):
        """Return the KeyMapping instance to be tested"""

    def test_accuracy(self):
        """Test the mapping on a large range of relative accuracies"""
        rel_acc_mult = 1 - math.sqrt(2) * 1e-1
        min_rel_acc = 1e-8
        rel_acc = 1 - 1e-3

        while rel_acc >= min_rel_acc:
            mapping = self.mapping(rel_acc, offset=0.0)
            max_rel_acc = _test_value_rel_acc(mapping)
            assert max_rel_acc < mapping.relative_accuracy
            rel_acc *= rel_acc_mult

    def test_offsets(self):
        """Test offsets"""
        for offset in self.offsets:
            mapping = self.mapping(0.01, offset=offset)
            assert mapping.key(1) == int(offset)


class TestLogarithmicMapping(BaseTestKeyMapping, TestCase):
    """Class for testing LogarithmicMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LogarithmicMapping(relative_accuracy, offset)


class TestLinearlyInterpolatedMapping(BaseTestKeyMapping, TestCase):
    """Class for testing LinearlyInterpolatedMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LinearlyInterpolatedMapping(relative_accuracy, offset)


class TestCubicallyInterpolatedMapping(BaseTestKeyMapping, TestCase):
    """Class for testing CubicallyInterpolatedMapping class"""  #

    def mapping(self, relative_accuracy, offset):
        return CubicallyInterpolatedMapping(relative_accuracy, offset)


@pytest.mark.parametrize("x", [-12.3, -1.0, -1.0 / 3.0, 0.0, 1.0, 1.0 / 3.0, 2.0 ** 10])
def test_cbrt(x):
    assert pytest.approx(_cbrt(x), numpy.cbrt(x))
