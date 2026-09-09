from test8_worker import compute_weighted_average
import pytest

def test_standard():
    assert compute_weighted_average([80, 90], [1, 1]) == 85.0

def test_zero_total_weights():
    # Must handle total zero weights gracefully without ZeroDivisionError
    assert compute_weighted_average([100, 50], [0, 0]) == 0.0

def test_string_numeric_inputs():
    # Must coerce string numbers gracefully
    assert compute_weighted_average(["70.0", "90.0"], ["1", "3"]) == 85.0

def test_empty_lists():
    assert compute_weighted_average([], []) == 0.0
