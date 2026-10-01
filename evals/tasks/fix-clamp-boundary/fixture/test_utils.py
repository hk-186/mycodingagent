# -*- coding: utf-8 -*-
import pytest

from utils import clamp


def test_within_range():
    assert clamp(5, 1, 10) == 5


def test_below_low_returns_low():
    assert clamp(0, 1, 10) == 1


def test_above_high_returns_high():
    assert clamp(11, 1, 10) == 10


def test_equal_boundaries():
    assert clamp(1, 1, 10) == 1
    assert clamp(10, 1, 10) == 10


def test_inverted_range_raises():
    with pytest.raises(ValueError):
        clamp(5, 10, 1)
