import abc
from unittest import TestCase

import pytest
import six

from ddsketch.mapping import CubicallyInterpolatedMapping
from ddsketch.mapping import LinearlyInterpolatedMapping
from ddsketch.mapping import LogarithmicMapping
from ddsketch.pb.proto import DDSketchProto
from ddsketch.pb.proto import KeyMappingProto
from ddsketch.pb.proto import StoreProto
from ddsketch.store import DenseStore
from tests.test_ddsketch import TestDDSketch
from tests.test_store import TestDenseStore


class BaseTestKeyMappingProto(six.with_metaclass(abc.ABCMeta)):
    offsets = [0, 1, -12.23, 7768.3]

    def test_round_trip(self):
        rel_accs = [1e-1, 1e-2, 1e-8]
        for rel_acc in rel_accs:
            for offset in self.offsets:
                mapping = self.mapping(rel_acc, offset)
                round_trip_mapping = KeyMappingProto.from_proto(
                    KeyMappingProto.to_proto(mapping)
                )
                assert type(mapping) == type(round_trip_mapping)
                assert mapping.relative_accuracy == pytest.approx(
                    round_trip_mapping.relative_accuracy
                )
                assert mapping.value(0) == pytest.approx(round_trip_mapping.value(0))


class TestLogarithmicMapping(BaseTestKeyMappingProto, TestCase):
    """Class for testing LogarithmicMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LogarithmicMapping(relative_accuracy, offset)


class TestLinearlyInterpolatedMapping(BaseTestKeyMappingProto, TestCase):
    """Class for testing LinearlyInterpolatedMapping class"""

    def mapping(self, relative_accuracy, offset):
        return LinearlyInterpolatedMapping(relative_accuracy, offset)


class TestCubicallyInterpolatedMapping(BaseTestKeyMappingProto, TestCase):
    """Class for testing CubicallyInterpolatedMapping class"""

    def mapping(self, relative_accuracy, offset):
        return CubicallyInterpolatedMapping(relative_accuracy, offset)


class TestStoreProto(TestDenseStore, TestCase):
    def _test_store(self, values):
        store = DenseStore()
        for val in values:
            store.add(val)
        self._test_values(StoreProto.from_proto(StoreProto.to_proto(store)), values)


class TestDDSketchProto(TestDDSketch, TestCase):
    def _evaluate_sketch_accuracy(self, sketch, data, eps, summary_stats=False):
        round_trip_sketch = DDSketchProto.from_proto(DDSketchProto.to_proto(sketch))
        super(TestDDSketchProto, self)._evaluate_sketch_accuracy(
            round_trip_sketch, data, eps, summary_stats
        )

    def test_add_multiple(self):
        """Override."""

    def test_add_decimal(self):
        """Override."""

    def test_merge_equal(self):
        """Override."""

    def test_merge_unequal(self):
        """Override."""

    def test_merge_mixed(self):
        """Override."""

    def test_consistent_merge(self):
        """Override."""
