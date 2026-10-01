# -*- coding: utf-8 -*-
from text_tools import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_consecutive_spaces_collapse():
    assert slugify("hello   world") == "hello-world"


def test_leading_trailing_separators_stripped():
    assert slugify("  Hello World  ") == "hello-world"


def test_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_mixed():
    assert slugify("  The  quick -- BROWN fox! ") == "the-quick-brown-fox"
