"""
calculator.py —— 一个简单的 Python 计算器

功能：
    1. 四则运算：+ - * /（支持小数、空格随意）
    2. 括号：( ) 可以任意嵌套
    3. 一元正负号：-3 + 5、2 * (-3)、-(1 + 2)
    4. 友好的错误提示：括号不匹配、除数为 0、非法字符等

实现思路：
    先做"词法分析"把字符串切成 token，再用"递归下降"方式按优先级解析：
        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := ('+' | '-') factor | number | '(' expr ')'

用法：
    交互模式：python calculator.py
    表达式模式：python calculator.py "1 + 2 * (3 - 4 / 2)"
"""

import sys


class CalculatorError(Exception):
    """计算器自身的错误（输入非法、除零、括号不匹配等）"""

# --------------------------------------------------------------------------
# 词法分析：把字符串切成 token 列表
# --------------------------------------------------------------------------
def tokenize(text: str):
    """把表达式字符串转换成 token 列表。

    token 形式为 (类型, 值)，类型有 NUMBER / OP / LPAREN / RPAREN。
    """
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():                       # 跳过空白
            i += 1
            continue
        if ch.isdigit() or ch == ".":          # 数字（含小数点）
            start = i
            dot_count = 0
            while i < n and (text[i].isdigit() or text[i] == "."):
                if text[i] == ".":
                    dot_count += 1
                    if dot_count > 1:
                        raise CalculatorError(f"数字格式错误：{text[start:i + 1]}")
                i += 1
            num_text = text[start:i]
            if num_text == ".":
                raise CalculatorError("数字格式错误：单独的小数点")
            tokens.append(("NUMBER", float(num_text)))
            continue
        if ch in "+-*/":
            tokens.append(("OP", ch))
            i += 1
            continue
        if ch == "(":
            tokens.append(("LPAREN", ch))
            i += 1
            continue
        if ch == ")":
            tokens.append(("RPAREN", ch))
            i += 1
            continue
        raise CalculatorError(f"无法识别的字符：{ch!r}")
    return tokens


# --------------------------------------------------------------------------
# 语法分析 + 求值：递归下降
# --------------------------------------------------------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    # ---- 工具方法 ----
    def peek(self):
        """看当前 token，没有则返回 None"""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        """取出当前 token 并后移"""
        token = self.peek()
        self.pos += 1
        return token

    # ---- 各层语法规则 ----
    def parse_expr(self):
        """expr := term (('+' | '-') term)*"""
        value = self.parse_term()
        while True:
            token = self.peek()
            if token and token[0] == "OP" and token[1] in "+-":
                op = self.advance()[1]
                rhs = self.parse_term()
                value = value + rhs if op == "+" else value - rhs
            else:
                break
        return value

    def parse_term(self):
        """term := factor (('*' | '/') factor)*"""
        value = self.parse_factor()
        while True:
            token = self.peek()
            if token and token[0] == "OP" and token[1] in "*/":
                op = self.advance()[1]
                rhs = self.parse_factor()
                if op == "*":
                    value = value * rhs
                else:
                    if rhs == 0:
                        raise CalculatorError("除数不能为 0")
                    value = value / rhs
            else:
                break
        return value

    def parse_factor(self):
        """factor := ('+' | '-') factor | number | '(' expr ')'"""
        token = self.peek()
        if token is None:
            raise CalculatorError("表达式不完整（缺少数字或括号）")

        kind, value = token
        if kind == "OP" and value in "+-":        # 一元正负号
            self.advance()
            operand = self.parse_factor()
            return operand if value == "+" else -operand
        if kind == "NUMBER":
            self.advance()
            return value
        if kind == "LPAREN":
            self.advance()
            inner = self.parse_expr()
            nxt = self.peek()
            if nxt is None or nxt[0] != "RPAREN":
                raise CalculatorError("括号不匹配（缺少右括号 ')'）")
            self.advance()
            return inner
        if kind == "RPAREN":
            raise CalculatorError("括号不匹配（多余的右括号 ')'）")
        raise CalculatorError(f"无法解析的 token：{value!r}")


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------
def calculate(expression: str) -> float:
    """计算表达式并返回结果（浮点数）。

    >>> calculate("1 + 2 * 3")
    7.0
    >>> calculate("(1 + 2) * 3")
    9.0
    """
    if expression is None or not expression.strip():
        raise CalculatorError("表达式不能为空")
    tokens = tokenize(expression)
    if not tokens:
        raise CalculatorError("表达式不能为空")
    parser = Parser(tokens)
    result = parser.parse_expr()
    if parser.pos != len(parser.tokens):        # 还有剩余 token，说明输入非法
        rest = parser.tokens[parser.pos]
        raise CalculatorError(f"表达式存在多余内容：{rest[1]!r}")
    return result


def format_result(value: float) -> str:
    """整数结果去掉末尾的 .0，更符合直觉；其余保留原样"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def main(argv) -> int:
    # 命令行带表达式：算一次就退出，方便脚本调用
    if len(argv) > 1:
        expression = " ".join(argv[1:])
        try:
            print(f"{expression} = {format_result(calculate(expression))}")
        except CalculatorError as e:
            print(f"错误：{e}", file=sys.stderr)
            return 1
        return 0

    # 交互模式
    print("简易计算器（支持 + - * / 和括号，输入 exit 或 quit 退出）")
    print("示例：(1 + 2) * 3 / 2   ->   4.5")
    while True:
        try:
            line = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            return 0

        if not line:
            continue
        if line.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            return 0
        try:
            print(f"= {format_result(calculate(line))}")
        except CalculatorError as e:
            print(f"错误：{e}")
        except ZeroDivisionError:
            print("错误：除数不能为 0")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
