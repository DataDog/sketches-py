# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

"""Tests for the KeyMapping classes"""

from abc import ABC, abstractmethod
import math
from unittest import TestCase

from ddsketch.mapping import (
    CubicallyInterpolatedMapping,
    KeyMapping,
    LogarithmicMapping,
    LinearlyInterpolatedMapping,
)


def _relative_error(expected_min, expected_max, actual):
    """ helper method to calculate the relative error """
    if expected_min < 0 or expected_max < 0 or actual < 0:
        raise Exception()
    if (expected_min <= actual) and (actual <= expected_max):
        return 0.0
    if expected_min == 0 and expected_max == 0:
        return 0.0 if actual == 0 else float("+inf")
    if actual < expected_min:
        return (expected_min - actual) / expected_min

    return (actual - expected_max) / expected_max


def _test_value_rel_acc(mapping, tester):
    """ calculate the relative accuracy of a mapping on a large range of values """
    value_mult = 2 - math.sqrt(2) * 1e-1
    max_relative_acc = 0.0
    value = mapping.min_possible
    while value < mapping.max_possible / value_mult:
        value *= value_mult
        map_val = mapping.value(mapping.key(value))
        rel_err = _relative_error(value, value, map_val)
        tester.assertLess(rel_err, mapping.relative_accuracy)
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


class TestKeyMapping(ABC):
    """Abstract class for testing KeyMapping classes"""

    offsets = [0, 1, -12.23, 7768.3]

    @abstractmethod
    def mapping(self, relative_accuracy, offset):
        """ return the KeyMapping instance to be tested """

    def test_accuracy(self):
        """ test the mapping on a large range of relative accuracies """
        rel_acc_mult = 1 - math.sqrt(2) * 1e-1
        min_rel_acc = 1e-8
        rel_acc = 1 - 1e-3

        while rel_acc >= min_rel_acc:
            mapping = self.mapping(rel_acc, offset=0.0)
            max_rel_acc = _test_value_rel_acc(mapping, self)
            self.assertLess(max_rel_acc, mapping.relative_accuracy)
            rel_acc *= rel_acc_mult

    def test_offsets(self):
        for offset in self.offsets:
            mapping = self.mapping(0.01, offset=offset)
            self.assertEqual(mapping.key(1), int(offset))

    def test_round_trip(self):
        rel_accs = [1e-1, 1e-2, 1e-8]
        for rel_acc in rel_accs:
            for offset in self.offsets:
                mapping = self.mapping(rel_acc, offset)
                round_trip_mapping = KeyMapping.from_proto(mapping.to_proto())
                self.assertEqual(type(mapping), type(round_trip_mapping))
                self.assertAlmostEqual(
                    mapping.relative_accuracy, round_trip_mapping.relative_accuracy
                )
                self.assertAlmostEqual(mapping.value(0), round_trip_mapping.value(0))


class TestLogarithmicMapping(TestKeyMapping, TestCase):
    """Class for testing LogarithmicMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LogarithmicMapping(relative_accuracy, offset)


class TestLinearlyInterpolatedMapping(TestKeyMapping, TestCase):
    """Class for testing LinearlyInterpolatedMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LinearlyInterpolatedMapping(relative_accuracy, offset)


class TestCubicallyInterpolatedMapping(TestKeyMapping, TestCase):
    """Class for testing CubicallyInterpolatedMapping class"""  #

    def mapping(self, relative_accuracy, offset):
        return CubicallyInterpolatedMapping(relative_accuracy, offset)
