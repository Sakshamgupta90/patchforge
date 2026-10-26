import pytest

from calc import add, divide


def test_add():
    assert add(2, 3) == 5


def test_divide():
    assert divide(9, 3) == 3


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
