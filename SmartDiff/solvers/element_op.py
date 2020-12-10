from SmartDiff.solvers.integrator import AutoDiff as AD
import numpy as np
from sympy import bell, symbols
from math import factorial # YW

# Here contains the derivative calculation of elementary operations
# If the input 'x' is a scalar number, it will return the value of such operation evaluated at 'x',
# which is directly applied the operation.
# If the input is a dual number(AD object), it will return another dual number(AD object),
# where the val represents the value evaluated at 'x', der represents the derivative to 'x' which evaluated at 'x'.
# derivatives for reference: http://math2.org/math/derivatives/tableof.htm

def get_n_der_vecs(dk_f, gx, N):
    """
    This function applies Faa di Bruno's formula to compute the derivatives of order from 1 to N
    given the calling elementary operator has dk_f as its kth order derivative function.

    :param dk_f(val, k): A lambda function of the kth order derivative of f at the point val
    :param gx: Potentially an AutoDiff object
    :param N: highest derivative order of gx

    :return: a list of high-order derivatives up until gx.N
    """
    # Create symbols and symbol-value mapping for eval() in the loop
    dxs = symbols('q:%d' % N)
    dx_mapping = {str(dxs[i]): gx.der[i] for i in range(N)}
    # Use Faa di Bruno's formula
    der_new = []
    for n in range(1, N + 1):
        nth_der = 0
        for k in range(1, n + 1):
            # The first n-k+1 derivatives
            t = n - k + 1
            vars = dxs[:t]
            # bell polynomial as python function str
            bell_nk_str = str(bell(n, k, vars))
            # evaluate the bell polynomial using the symbol-value mapping
            val_bell_nk = eval(bell_nk_str, dx_mapping)
            nth_der += dk_f(gx.val, k) * val_bell_nk
        der_new.append(nth_der)
    return der_new


def power_k_order(gx, n, k):
    """Returns the kth order derivative of gx^n
    """
    if type(n) != int and type(n) != float:
        raise AttributeError('Type error!')

    if n == 1/2 and gx < 0:
        raise ValueError('Error: Independent variable must be nonnegative!')

    if type(n) == int and n >= 0:
        if k <= n:
            falling = 1
            for i in range(k):
                falling *= n - i
            return falling * gx ** (n-k)
        if k > n:
            return 0
    else:
        falling = 1
        for i in range(k):
            falling *= n - i
        return falling * gx ** (n - k)


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
    N = 1
    dk_f = lambda gx, k: power_k_order(gx, n, k) # nth order derivative for power(x,n)
    try:
        val_new = np.power(x.val, n)
        N = x.N
        if N == 1:
            der_new = np.array([n * x.val ** (n - 1) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.power(x, n)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def log_k_order(gx, n, k):
    if k == 1:
        return 1/(gx * np.log(n))
    if k != 1:
        return power_k_order(gx, -1, k-1)/(np.log(n))

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
    N = 1
    dk_f = lambda gx, k: log_k_order(gx, n, k)  # nth order derivative for log(x, n)
    try:
        val_new = np.log(x.val) / np.log(n)
        N = x.N
        if N == 1:
            der_new = np.array([1 / (x.val * np.log(n) * x.der)])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.log(x) / np.log(n)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def expn(x,n):
    #  (n^{x})' = n^{x} * ln(n) * x'
    """Returns the value and derivative of a exponential operation: n^x

        INPUTS
        =======
        x: an AutoDiff object or a scalar, required, the input variable
        n: float or int, required, the base

        RETURNS
        ========
        an AD object containing the value and derivative of the expression

        EXAMPLES
        =========
        >>> exp(1.0, np.e)
        AD(2.718281828459045, 0)
        >>> exp(AD(1.0, 2.0), np.e)
        AD(2.718281828459045, 5.43656365691809)
        """
    N = 1
    dk_f = lambda gx, k: n**gx *(np.log(n)**2)  # nth order derivative for n^x
    try:
        val_new = n**x.val
        N = x.N
        if N == 1:
            der_new = np.array([n**x.val * np.log(n) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = n**x
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


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
    N = 1
    dk_f = lambda gx, k: np.exp(gx)  # nth order derivative for e^x
    try:
        val_new = np.exp(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([np.exp(x.val) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.exp(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


# YW copied from Xincheng's code on github
def inv(x):
    """
    Inverse of a term (x cannot be 0)
    :param x:
    :return:
    """
    N = 1
    dk_f = lambda gx, k: ((-1.0) ** k * factorial(k)) / (gx ** (k+1)) # YW
    try:
        val_new = 1.0 / x.val
        N = x.N
        if N == 1:
            der_new = -x.der / (x.val**2)
        else:
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x == 0:
                raise ZeroDivisionError('Division by zero')
            val_new = 1.0 / x
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N) # YW


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
    N = 1
    dk_f = lambda gx, k: power_k_order(gx, 1/2, k)  # nth order derivative for sqrt(x)
    try:
        val_new = np.sqrt(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([1 / 2 * x.val ** (-1 / 2) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.sqrt(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


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
    half_pi = np.pi / 2
    N = 1
    dk_f = lambda gx, k: np.sin(gx + half_pi * k)  # nth order derivative for sin(x)
    try:
        val_new = np.sin(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([np.cos(x.val) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.sin(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


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
    half_pi = np.pi / 2
    N = 1
    dk_f = lambda gx, k: np.cos(gx + half_pi * k)  # nth order derivative for cos(x)
    try:
        val_new = np.cos(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([-np.sin(x.val) * x.der])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.cos(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def tan_k_order(gx, k):
    if k == 1:
        return 1 / (np.cos(gx)) ** 2
    else:
        x = AD(gx, N=k-1)
        f = cos(x)
        dk_f2 = lambda gx2, k2: power_k_order(gx2, -2, k2)
        der_new = get_n_der_vecs(dk_f2, f, k-1)
        return np.float(der_new[k-2])

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
    N = 1
    dk_f = lambda gx, k: tan_k_order(gx, k)  # nth order derivative for tan(x)
    try:
        val_new = np.tan(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([1 / (np.cos(x.val)) ** 2])
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.tan(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def arcsin_k_order(gx, k):
    if k == 1:
        return 1 / np.sqrt(1 - gx**2)
    else:
        x = AD(gx, N=k-1)
        f = 1 - power(x,2)
        dk_f2 = lambda gx2, k2: power_k_order(gx2, -1/2, k2)
        der_new = get_n_der_vecs(dk_f2, f, k-1)
        return np.float(der_new[k-2])

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
    N = 1
    dk_f = lambda gx, k: arcsin_k_order(gx, k)  # nth order derivative for arcsin(x)
    if isinstance(x, AD):
        if x.val < -1 or x.val > 1:
            raise ValueError('Error: Independent variable must be in [-1,1]!')
    try:
        val_new = np.arcsin(x.val)
        N = x.N
        if N == 1:
            der_new = np.array(1 / np.sqrt(1 - x.val ** 2) * x.der)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < -1 or x > 1:
                raise ValueError('Error: Independent variable must be in [-1,1]!')
            val_new = np.arcsin(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


#### arccos is not correct for the 1th der when k>1
def arccos_k_order(gx, k):
    if k == 1:
        return - 1 / np.sqrt(1 - gx**2)
    else:
        x = AD(gx, N=k-1)
        f = 1 - power(x,2)
        dk_f2 = lambda gx2, k2: - power_k_order(gx2, -1/2, k2)
        der_new = get_n_der_vecs(dk_f2, f, k-1)
        return np.float(der_new[k-2])

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
    N = 1
    dk_f = lambda gx, k: arccos_k_order(gx, k)
    if isinstance(x, AD):
        if x.val < -1 or x.val > 1:
            raise ValueError('Error: Independent variable must be in [-1,1]!')
    try:
        val_new = np.arcsin(x.val)
        N = x.N
        if N == 1:
            der_new = np.array(-1 / np.sqrt(1 - x.val ** 2) * x.der)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < -1 or x > 1:
                raise ValueError('Error: Independent variable must be in [-1,1]!')
            val_new = np.arccos(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def arctan_k_order(gx, k):
    if k == 1:
        return 1 / (1 + gx ** 2)
    else:
        x = AD(gx, N = k-1)
        f = 1 + power(x,2)
        dk_f2 = lambda gx2, k2: power_k_order(gx2, -1.0, k2)
        der_new = get_n_der_vecs(dk_f2, f, k-1)
        return np.float(der_new[k-2])

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
    N = 1
    dk_f = lambda gx, k: arctan_k_order(gx, k)  # nth order derivative for arctan(x)
    try:
        val_new = np.arctan(x.val)
        N = x.N
        if N == 1:
            der_new = np.array(1 / (1 + x.val ** 2) * x.der)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.arctan(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def sinh_k_order(gx, k):
    if k%2 == 1:
        return np.cosh(gx)
    if k%2 == 0:
        return np.sinh(gx)

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
    N = 1
    dk_f = lambda gx, k: sinh_k_order(gx, k)  # nth order derivative for sinh(x)
    try:
        val_new = np.sinh(x.val)
        N = x.N
        if N == 1:
            der_new = np.array(np.cosh(x.val) * x.der)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.sinh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def cosh_k_order(gx, k):
    if k%2 == 1:
        return np.sinh(gx)
    if k%2 == 0:
        return np.cosh(gx)

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
    N = 1
    dk_f = lambda gx, k: cosh_k_order(gx, k)  # nth order derivative for cosh(x)
    try:
        val_new = np.cosh(x.val)
        N = x.N
        if N == 1:
            der_new = np.array(np.sinh(x.val) * x.der)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.cosh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)


def tanh(x):
    # (tanh(x))' = (1 - tanh(x)**2) * x'
    """Returns the value and derivative of a hyperbolic tangent operation: tanh(x)
    tanh(x)=(e^(2x)-1)/(e^(2x)+1), i.e. f(g(h(x))) with f(x)=(x-1)/(x+1) and g(x)=e^(2x)

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
    >>> tanh(AD(0.0, N=2.0))
    AD(0.0, [1., 0, -2.]) # YW maybe the data type isn't correct
    """
    N = 1
    try:
        val_new = np.tanh(x.val)
        N = x.N
        if N == 1:
            der_new = np.array([(1 - np.tanh(x.val) ** 2) * x.der])
        else:
            # N > 1
            f = lambda x: 1-2/(x+1)
            g = lambda x: exp(2*x)
            der_new = f(g(x)).der
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            val_new = np.tanh(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AD(val_new, der_new, N)
