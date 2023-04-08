import sympy
from sympy import lambdify
from sympy.abc import x

# see https://ethaneade.com/lie.pdf

def taylor_series_near_zero(x, f, order=6, eps=1e-7, verbose=False):
    """
    Create sympy function which gives numpy type data when evaluated
    """
    f_series = f.series(x, 0, order).removeO()
    s = (x)
    g = lambdify(s, f, modules='numpy')
    return g

#Define Taylor Series Approximation Function used in Ethan Eade's Lie Group
x = sympy.symbols("x")
series_dict = {}
series_dict["sin(x)/x"] = taylor_series_near_zero(x, sympy.sin(x) / x)
series_dict["(1 - cos(x))/x"] = taylor_series_near_zero(x, (1 - sympy.cos(x)) / x)
series_dict["(1 - cos(x))/x^2"] = taylor_series_near_zero(x, (1 - sympy.cos(x)) / x**2)
series_dict["(1 - sin(x))/x^3"] = taylor_series_near_zero(x, (1 - sympy.sin(x)) / x**3)
