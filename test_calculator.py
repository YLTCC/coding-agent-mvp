"""
test_calculator.py —— calculator.py 的测试

运行方式（二选一）：
    pytest test_calculator.py
    python test_calculator.py     （不依赖 pytest，直接跑断言）
"""

from calculator import CalculatorError, calculate, format_result


def test_basic_operations():
    assert calculate("1 + 2") == 3
    assert calculate("7 - 10") == -3
    assert calculate("6 * 7") == 42
    assert calculate("8 / 2") == 4
    assert calculate("1 / 2") == 0.5


def test_priority():
    assert calculate("1 + 2 * 3") == 7
    assert calculate("2 - 3 * 4") == -10
    assert calculate("10 - 4 / 2") == 8
    assert calculate("2 * 3 + 4 * 5") == 26


def test_parentheses():
    assert calculate("(1 + 2) * 3") == 9
    assert calculate("2 * (3 + (4 - 1))") == 12
    assert calculate("((2 + 3) * (4 - 1)) / 5") == 3
    assert calculate("10 / (2 + 3)") == 2


def test_unary_and_spaces():
    assert calculate("-3 + 5") == 2
    assert calculate("2 * -3") == -6
    assert calculate("-(1 + 2)") == -3
    assert calculate("+4 - -4") == 8
    assert calculate("  1   +   2 * 3  ") == 7


def test_decimals():
    assert calculate("0.5 + 0.25") == 0.75
    assert calculate("1.5 * 2") == 3
    assert calculate("(.5 + .5)") == 1


def test_errors():
    for bad in ["", "   ", "1 +", "1 + * 2", "(1 + 2", "1 + 2)", "1 $ 2",
                "()", "1.2.3"]:
        try:
            calculate(bad)
        except CalculatorError:
            pass
        else:
            raise AssertionError("应当报错但没有：%r" % bad)

    try:
        calculate("1 / 0")
    except CalculatorError as e:
        assert "0" in str(e)
    else:
        raise AssertionError("除以 0 应当报错")


def test_format_result():
    assert format_result(6.0) == "6"
    assert format_result(4.5) == "4.5"
    assert format_result(-3.0) == "-3"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("[PASS] %s" % fn.__name__)
    print("\n全部 %d 组测试通过！" % len(tests))
