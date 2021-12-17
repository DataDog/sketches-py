# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache License 2.0.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2020 Datadog, Inc.

import abc

import numpy as np
import six


class Dataset(six.with_metaclass(abc.ABCMeta)):
    def __init__(self, size):
        self.size = int(size)
        self.data = self.populate()

    def __str__(self):
        return "{}_{}".format(self.name, self.size)

    def __len__(self):
        return self.size

    def rank(self, value):
        lower = np.array(sorted(self.data)) < value
        if np.all(lower):
            return self.size - 1
        else:
            return np.argmin(lower)

    def quantile(self, q):
        self.data.sort()
        rank = int(q * (self.size - 1))
        return self.data[rank]

    @property
    def sum(self):  # noqa: A003
        return np.sum(self.data)

    @property
    def avg(self):
        return np.mean(self.data)

    @abc.abstractmethod
    def name(self):
        """Name of dataset"""

    @abc.abstractmethod
    def populate(self):
        """Populate self.data with self.size values"""


class EmptyDataset(Dataset):
    @property
    def name(self):
        return "no_name"

    def populate(self):
        return []

    def add(self, val):
        self.size += 1
        self.data.append(val)

    def add_all(self, vals):
        self.size += len(vals)
        self.data.extend(vals)


class UniformForward(Dataset):
    @property
    def name(self):
        return "uniform_forward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(self.size):
            yield x


class UniformBackward(Dataset):
    @property
    def name(self):
        return "uniform_backward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(self.size, 0, -1):
            yield x


class NegativeUniformForward(Dataset):
    @property
    def name(self):
        return "negative_uniform_forward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(self.size, 0, -1):
            yield -x


class NegativeUniformBackward(Dataset):
    @property
    def name(self):
        return "negative_uniform_backward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(self.size):
            yield -x


class NumberLineForward(Dataset):
    @property
    def name(self):
        return "number_line_forward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(-self.size // 2 + 1, self.size // 2 + 1, 1):
            yield x


class NumberLineBackward(Dataset):
    @property
    def name(self):
        return "number_line_backward"

    def populate(self):
        return list(self.generate())

    def generate(self):
        for x in range(self.size // 2, -self.size // 2, -1):
            yield x


class UniformZoomIn(Dataset):
    @property
    def name(self):
        return "uniform_zoomin"

    def populate(self):
        return list(self.generate())

    def generate(self):
        if self.size % 2 == 1:
            for item in range(self.size // 2):
                yield item
                yield self.size - item - 1
            yield self.size // 2
        else:
            for item in range(self.size // 2):
                yield item
                yield self.size - item - 1


class UniformZoomOut(Dataset):
    @property
    def name(self):
        return "uniform_zoomout"

    def populate(self):
        return list(self.generate())

    def generate(self):
        if self.size % 2 == 1:
            yield self.size // 2
            half = int(np.floor(self.size / 2))
            for item in range(1, half + 1):
                yield half + item
                yield half - item
        else:
            half = int(np.ceil(self.size / 2)) - 0.5
            for item in range(0, int(half + 0.5)):
                yield int(half + item + 0.5)
                yield int(half - item - 0.5)


class UniformSqrt(Dataset):
    @property
    def name(self):
        return "uniform_sqrt"

    def populate(self):
        return list(self.generate())

    def generate(self):
        t = int(np.sqrt(2 * self.size))
        initial_item = 0
        initial_skip = 1
        emitted = 0
        i = 0
        while emitted < self.size:
            item = initial_item
            skip = initial_skip
            for j in range(t - i):
                if item < self.size:
                    yield item
                    emitted += 1
                item += skip
                skip += 1
            if t - i > 1:
                initial_skip += 1
                initial_item += initial_skip
                i += 1
            else:
                initial_item += 1


class Constant(Dataset):

    constant = 42.0

    @property
    def name(self):
        return "constant"

    def populate(self):
        return [self.constant] * self.size


class Exponential(Dataset):

    scale = 0.01

    @classmethod
    def from_params(cls, scale, n):
        cls.scale = scale
        return cls(n)

    @property
    def name(self):
        return "exponential"

    def populate(self):
        return np.random.exponential(scale=self.scale, size=self.size)


class Lognormal(Dataset):

    scale = 100.0

    @classmethod
    def from_params(cls, scale, n):
        cls.scale = scale
        return cls(n)

    @property
    def name(self):
        return "lognormal"

    def populate(self):
        return np.random.lognormal(size=self.size) / self.scale


class Normal(Dataset):

    loc = 37.4
    scale = 1.0

    @classmethod
    def from_params(cls, loc, scale, n):
        cls.loc = loc
        cls.scale = scale
        return cls(n)

    @property
    def name(self):
        return "normal"

    def populate(self):
        return np.random.normal(loc=self.loc, scale=self.scale, size=self.size)


class Laplace(Dataset):

    loc = 11278.0
    scale = 100.0

    @classmethod
    def from_params(cls, loc, scale, n):
        cls.loc = loc
        cls.scale = scale
        return cls(n)

    @property
    def name(self):
        return "laplace"

    def populate(self):
        return np.random.laplace(loc=self.loc, scale=self.scale, size=self.size)


class Bimodal(Dataset):

    right_loc = 17.3
    left_loc = -2.0
    left_std = 3.0

    @property
    def name(self):
        return "bimodal"

    def populate(self):
        return [next(self.generate()) for _ in range(int(self.size))]

    def generate(self):
        if np.random.random() > 0.5:
            yield np.random.laplace(self.right_loc)
        else:
            yield np.random.normal(self.left_loc, self.left_std)


class Mixed(Dataset):

    mean = 0.0
    sigma = 0.25
    scale_factor = 0.1

    loc = 10.0
    scale = 0.5

    def __init__(self, size, ratio=0.9, ignore_rank=False):
        self.size = int(size)
        self.ratio = ratio
        self.data = self.populate()
        self._ignore_rank = ignore_rank

    @property
    def name(self):
        return "mixed"

    def populate(self):
        return [next(self.generate()) for _ in range(int(self.size))]

    def generate(self):
        if np.random.random() < self.ratio:
            yield self.scale_factor * np.random.lognormal(self.mean, self.sigma)
        else:
            yield np.random.normal(self.loc, self.scale)


class Trimodal(Dataset):

    right_loc = 17.3
    left_loc = 5.0
    left_std = 0.5
    exp_scale = 0.01

    @property
    def name(self):
        return "trimodal"

    def populate(self):
        return [next(self.generate()) for _ in range(int(self.size))]

    def generate(self):
        if np.random.random() > 2.0 / 3.0:
            yield np.random.laplace(self.right_loc)
        elif np.random.random() > 1.0 / 3.0:
            yield np.random.normal(self.left_loc, self.left_std)
        else:
            yield np.random.exponential(scale=self.exp_scale)


class Integers(Dataset):

    loc = 4.3
    scale = 5.0

    @classmethod
    def from_params(cls, loc, scale, n):
        cls.loc = loc
        cls.scale = scale
        return cls(n)

    @property
    def name(self):
        return "integers"

    def populate(self):
        return [
            int(x)
            for x in np.random.normal(loc=self.loc, scale=self.scale, size=self.size)
        ]
