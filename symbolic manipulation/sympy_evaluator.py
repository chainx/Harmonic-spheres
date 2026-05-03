import sympy as sp


def diff_expr(sympy_expr, diff_symbol):
    """
    Symbolically differentiate a SymPy expression.

    Parameters
    ----------
    sympy_expr : sympy.Expr
        The symbolic expression to differentiate.
    diff_symbol : sympy.Symbol
        The symbol to differentiate with respect to.

    Returns
    -------
    sympy.Expr
        The symbolic derivative of sympy_expr with respect to diff_symbol.
    """
    return sp.diff(sympy_expr, diff_symbol)


def eval_expr(sympy_expr, values_dict):
    """
    Evaluate a SymPy expression after substituting numerical values.

    Parameters
    ----------
    sympy_expr : sympy.Expr
        The symbolic expression to evaluate.
    values_dict : dict
        Dictionary mapping SymPy symbols to numerical values.

    Returns
    -------
    float or complex
        Numerical value of the expression.
    """
    substituted = sympy_expr.subs(values_dict)
    numerical = sp.N(substituted)

    if numerical.is_real:
        return float(numerical)
    return complex(numerical)
