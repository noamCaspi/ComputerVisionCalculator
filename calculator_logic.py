
class CalculatorLogic:

    DIGIT = "digit"
    OP = "op"
    CLEAR = "clear"
    EQUALS = "equals"

    def __init__(self):
        self._do_clear()

    def get_display(self) -> str:
        return self._current_display

    def type_in(self, c: str) -> None:
        if c.isdigit() or c == ".":
            self._do_digit_clicked(c)
        elif c == "=":
            self._do_equals()
        elif c == "C":
            self._do_clear()
        elif c in {"*", "+", "-", "/"}:
            self._do_math_op(c)
        else:
            raise ValueError("Unknown key")

    def _do_clear(self) -> None:
        self._cur_num = "0"
        self._current_display = "0"
        self._op = "nop"
        self._prev_result = float(0)
        self._last_clicked = self.CLEAR

    def _do_equals(self) -> None:
        func = self._get_op_function(self._op)
        try:
            self._prev_result = func(float(self._prev_result), float(self._cur_num))
            self._set_display(self._prev_result)
        except ZeroDivisionError:
            self._prev_result = 0
            self._current_display = "Nan"
        self._last_clicked = self.EQUALS

    def _do_math_op(self, op: str) -> None:
        if self._last_clicked != self.OP and self._last_clicked != self.EQUALS:
            self._do_equals()
        self._op = op
        self._last_clicked = self.OP

    def _do_digit_clicked(self, digit: str) -> None:
        if self._last_clicked is not self.DIGIT:
            self._cur_num = digit
        else:
            if digit != "." or "." not in self._cur_num:
                self._cur_num += digit
        self._set_display(float("0" + self._cur_num))
        if self._last_clicked == self.EQUALS:
            self._op = "nop"
        self._last_clicked = self.DIGIT

    def _set_display(self, num: float) -> None:
        if num.is_integer():
            self._current_display = str(int(num))
        else:
            self._current_display = str(num)

    @staticmethod
    def _get_op_function(action: str):
        if action == "+":
            return lambda x, y: x + y
        elif action == "*":
            return lambda x, y: x * y
        elif action == "/":
            return lambda x, y: x / y
        elif action == "-":
            return lambda x, y: x - y
        elif action == "nop":
            return lambda x, y: y
        else:
            raise ValueError("Unknown operator: " + action)
