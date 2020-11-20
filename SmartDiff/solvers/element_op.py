from SmartDiff.solvers.integrator import AutoDiffToy as AD
import numpy as np

# Here contains the derivative calculation of elementary operations
# If the input 'x' is a scalar number, it will return the value of such operation evaluated at 'x',
# which is directly applied the operation.
# If the input is a dual number(AD object), it will return another dual number(AD object),
# where the val represents the value evaluated at 'x', der represents the derivative to 'x' which evaluated at 'x'.
# derivatives for reference: http://math2.org/math/derivatives/tableof.htm

def power(x, n):
    # ((x)^n)' = n * (x)^{n-1} * x'
    """Returns the value and derivative of a power operation: x^n

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable
    n: float or int, required, the base

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> power(1.0, 2.0)
    AD(1.0, 0)
    >>> power(AD(1.0, 2.0), 2.0)
    AD(1.0, 4.0)
    """
    try:
        val_new = np.power(x.val, n)
        der_new = n * x.val ** (n - 1) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.power(x, n)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def log(x, n):
    # (log_n(x))' = 1/(x * log_e(n) * x')
    # we should also check the value >0 for log calculation
    """Returns the value and derivative of a logarithm operation: log_n(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable
    a: float or int, required, the base

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> log(np.e, np.e)
    AD(1.0, 0)
    >>> log(AD(np.e**2, 2.0), np.e)
    AD(2.0, 0.06766764161830635)
    """
    if isinstance(x, AD):
        if x.val <= 0:
            raise ValueError('Error: Independent variable must be positive!')
    try:
        val_new = np.log(x.val) / np.log(n)
        der_new = 1 / (x.val * np.log(n) * x.der)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x <= 0:
                raise ValueError('Error: Independent variable must be positive!')
            val_new = np.log(x) / np.log(n)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def exp(x):
    # (e^{x})' = e^{x} * x'
    """Returns the value and derivative of a exponential operation: e^x

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> exp(1.0)
    AD(2.718281828459045, 0)
    >>> exp(AD(1.0, 2.0))
    AD(2.718281828459045, 5.43656365691809)
    """
    try:
        val_new = np.exp(x.val)
        der_new = np.exp(x.val) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.exp(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def sqrt(x):
    # (sqrt(x))' = ((x)^{1/2})' = 1/2 * (x)^{-1/2} * x'
    # we should also check the value is >0 for sqrt calculation
    """Returns the value and derivative of a square root operation: x^{1/2}

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> sqrt(1.0)
    AD(1.0, 0)
    >>> sqrt(AD(1.0, 2.0))
    AD(1.0, 1.0)
    """
    if isinstance(x, AD):
        if x.val < 0:
            raise ValueError('Error: Independent variable must be nonnegative!')
    try:
        val_new = np.sqrt(x.val)
        der_new = 1 / 2 * x.val ** (-1 / 2) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < 0:
                raise ValueError('Error: Independent variable must be nonnegative!')
            val_new = np.sqrt(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def sin(x):
    # (sin(x))' = cos(x) * x'
    """Returns the value and derivative of a sine operation: sin(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> sin(0.0)
    AD(0.0, 0)
    >>> sin(AD(0.0, 2.0))
    AD(0.0, 2.0)
    """
    try:
        val_new = np.sin(x.val)
        der_new = np.cos(x.val) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.sin(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def cos(x):
    # (cos(x))' = - sin(x) * x'
    """Returns the value and derivative of a cosine operation: cos(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> cos(0.0)
    AD(1.0, 0)
    >>> cos(AD(0.0, 2.0))
    AD(1.0, -0.0)
    """
    try:
        val_new = np.cos(x.val)
        der_new = - np.sin(x.val) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.cos(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def tan(x):
    # (tan(x))' = 1/cos(x)^2 * x'
    """Returns the value and derivative of a tangent operation: tan(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> tan(0.0)
    AD(0.0, 0)
    >>> tan(AD(0.0, 2.0))
    AD(0.0, 2.0)
    """
    try:
        val_new = np.tan(x.val)
        der_new = 1 / (np.cos(x.val)) ** 2 * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.tan(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def arcsin(x):
    # (arcsin(x))' = 1/sqrt(1-(x)^2) * x'
    """Returns the value and derivative of an arcsine operation: arcsin(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> arcsin(0.0)
    AD(0.0, 0)
    >>> arcsin(AD(0.0, 2.0))
    AD(0.0, 0.0)
    """
    if isinstance(x, AD):
        if x.val < -1 or x.val > 1:
            raise ValueError('Error: Independent variable must be in [-1,1]!')
    try:
        val_new = np.arcsin(x.val)
        der_new = 1 / np.sqrt(1 - x.val ** 2) * x.val
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < -1 or x > 1:
                raise ValueError('Error: Independent variable must be in [-1,1]!')
            val_new = np.arcsin(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def arccos(x):
    # (arccos(x))' = - 1/sqrt(1-(x)^2) * x'
    """Returns the value and derivative of an arccosine operation: arccos(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> arccos(0.0)
    AD(1.5707963267948966, 0)
    >>> arccos(AD(0.0, 2.0))
    AD(1.5707963267948966, -0.0)
    """
    if isinstance(x, AD):
        if x.val < -1 or x.val > 1:
            raise ValueError('Error: Independent variable must be in [-1,1]!')
    try:
        val_new = np.arccos(x.val)
        der_new = -1 / np.sqrt(1 - x.val ** 2) * x.val
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < -1 or x > 1:
                raise ValueError('Error: Independent variable must be in [-1,1]!')
            val_new = np.arccos(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def arctan(x):
    # (arctan(x))' = 1/(1+(x)**2) * x'
    """Returns the value and derivative of an arctangent operation: arctan(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> arctan(0.0)
    AD(0.0, 0)
    >>> arctan(AD(0.0, 2.0))
    AD(0.0, 2.0)
    """
    try:
        val_new = np.arctan(x.val)
        der_new = 1 / (1 + x.val ** 2) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.arctan(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def sinh(x):
    # (sinh(x))' = cosh(x) * x'
    """Returns the value and derivative of a hyperbolic sine operation: sinh(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> sinh(0.0)
    AD(0.0, 0)
    >>> sinh(AD(0.0, 2.0))
    AD(0.0, 2.0)
    """
    try:
        val_new = np.sinh(x.val)
        der_new = np.cosh(x.val) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.sinh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def cosh(x):
    # (cosh(x))' = sinh(x) * x'
    """Returns the value and derivative of a hyperbolic cosine operation: cosh(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> cosh(0.0)
    AD(1.0, 0)
    >>> cosh(AD(0.0, 2.0))
    AD(1.0, 0.0)
    """
    try:
        val_new = np.cosh(x.val)
        der_new = np.sinh(x.val) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.cosh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)


def tanh(x):
    # (tanh(x))' = (1 - tanh(x)**2) * x'
    """Returns the value and derivative of a hyperbolic tangent operation: tanh(x)

    INPUTS
    =======
    x: an AutoDiff object or a scalar, required, the input variable

    RETURNS
    ========
    an AD object containing the value and derivative of the expression

    EXAMPLES
    =========
    >>> tanh(0.0)
    AD(0.0, 0)
    >>> tanh(AD(0.0, 2.0))
    AD(0.0, 2.0)
    """

    try:
        val_new = np.tanh(x.val)
        der_new = (1 - np.tanh(x.val) ** 2) * x.der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.tanh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = 0
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new)