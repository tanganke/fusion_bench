class pattern_query:
    R"""
    Examples:

        >>> f = pattern_query(lambda x: x==1)
        >>> f(1)
        True
        >>> f(2)
        False

    """

    def __init__(self, func=None, type=None):
        self.func = func
        self.type = type

    def __call__(self, expr) -> bool:
        if self.type is not None:
            if not isinstance(expr, self.type):
                return False
        if self.func is not None:
            return self.func(expr)
        return True


def is_expr_match(pattern, expr):
    R"""match pattern with a python expression expr.

    Examples:

        >>> is_expr_match('a', 'a')
        True
        >>> is_expr_match((object, 1), ('s',1))
        True
        >>> is_expr_match((object, 1), ('s',2))
        False
        >>> is_expr_match(((int, (int,)), (int, (int,)), (-1,)),
                          ((2146, (6,)), (1124, (97,)), (-1,)))
        True

        match a numpy array whose shape is (1,2)

        >>> import numpy as np
        >>> is_expr_match(
                pattern_query(lambda arr: arr.shape==(1,2), np.ndarray),
                np.zeros((1,2)))
        True

    Args:
        pattern: pattern to match
        expr:  python object

    Raises:
        NotImplementedError: Unsupported type

    Returns:
        bool
    """

    if type(pattern) == type:  # type
        if not isinstance(expr, pattern):
            return False

    else:  # instance
        if isinstance(pattern, pattern_query):
            return pattern(expr)
        if type(pattern) != type(expr):
            return False

        if isinstance(pattern, (int, float, str)):
            if pattern != expr:
                return False
        elif isinstance(pattern, (tuple, list)):
            if len(pattern) != len(expr):
                return False
            for i in range(len(pattern)):
                if not is_expr_match(pattern[i], expr[i]):
                    return False
        elif isinstance(pattern, dict):
            if len(pattern) != len(expr):
                return False
            for k in pattern:
                try:
                    if not is_expr_match(pattern[k], expr[k]):
                        return False
                except:
                    return False
        else:
            raise NotImplementedError("Unsupported type: {}".format(type(pattern)))
    return True
