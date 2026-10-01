# -*- coding: utf-8 -*-
from string_utils import is_palindrome, shout


def test_shout():
    assert shout("hi") == "HI"


def test_simple_palindrome():
    assert is_palindrome("level")


def test_not_palindrome():
    assert not is_palindrome("hello")


def test_ignore_case():
    assert is_palindrome("Level")


def test_ignore_non_alnum():
    assert is_palindrome("A man, a plan, a canal: Panama")


def test_empty_and_single():
    assert is_palindrome("")
    assert is_palindrome("x")
