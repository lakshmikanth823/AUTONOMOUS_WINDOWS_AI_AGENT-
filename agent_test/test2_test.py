from test2_discount import calculate_discounted_price
import pytest

def test_standard_discount():
    assert calculate_discounted_price(100.0, 20.0) == 80.0

def test_rounding():
    assert calculate_discounted_price(99.99, 15.0) == 84.99

def test_zero_discount():
    assert calculate_discounted_price(50.0, 0.0) == 50.0

def test_invalid_negative_discount():
    with pytest.raises(ValueError):
        calculate_discounted_price(100.0, -10.0)

def test_invalid_over_hundred_discount():
    with pytest.raises(ValueError):
        calculate_discounted_price(100.0, 110.0)
