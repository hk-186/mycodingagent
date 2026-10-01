# -*- coding: utf-8 -*-
import pathlib

import pytest

from csv_stats import column_average, read_rows


@pytest.fixture
def sample_csv():
    return str(pathlib.Path(__file__).parent / "sample.csv")


def test_read_rows(sample_csv):
    rows = read_rows(sample_csv)
    assert len(rows) == 5
    assert rows[0]["name"] == "apple"


def test_column_average(sample_csv):
    # price 列：2.5 / 3.0 / 2.6 / 空 / 2.7 -> 平均 2.7
    assert column_average(sample_csv, "price") == pytest.approx(2.7)


def test_column_average_skips_bad_values(sample_csv):
    # amount 列：10 / 20 / abc / 30 / 空 -> 仅 10/20/30 有效，平均 20.0
    assert column_average(sample_csv, "amount") == pytest.approx(20.0)


def test_missing_column_raises(sample_csv):
    with pytest.raises(KeyError):
        column_average(sample_csv, "nope")


def test_no_valid_numbers_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("name,price\napple,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        column_average(p, "price")
