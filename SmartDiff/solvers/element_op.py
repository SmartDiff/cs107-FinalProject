
import numpy as np
from sympy import bell, binomial, symbols
from math import factorial

# Here contains the derivative calculation of elementary operations
# If the input 'x' is a scalar number, it will return the value of such operation evaluated at 'x',
# which is directly applied the operation.
# If the input is a dual number(AD object), it will return another dual number(AD object),
# where the val represents the value evaluated at 'x', der represents the derivative to 'x' which evaluated at 'x'.
# derivatives for reference: http://math2.org/math/derivatives/tableof.html


class AutoDiff():
    def __init__(self, value, der=None, N=1, signatures=None, ):
        self.val = value
        self.N = N
        if der is None:
            self.der = np.zeros(N)
            self.der[0] = 1.0
        else:
            self.der = np.array(der)

    def __add__(self, other):
        # (f+g)' = f' + g'
        # (f+g)^(n) = f^(n) + g^(n)
        try:
            # elementwise summation to return Jacobian matrix
            val_new = self.val + other.val
            N_new = self.N  # The larger order of the two components
            if self.N == other.N:
                der_new = self.der + other.der
            elif self.N > other.N:  # other is constant
                der_new = self.der
            else:  # self is constant
                N_new = other.N
                der_new = other.der
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val + other
                der_new = self.der
                N_new = self.N
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new, N_new)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # (f-g)' = f' - g'
        # (f-g)^(n) = f^(n) - g^(n)
        try:
            # elementwise subtraction to return Jacobian matrix
            val_new = self.val - other.val
            N_new = self.N  # The larger order of the two components
            if self.N == other.N:
                der_new = self.der - other.der
            elif self.N > other.N:  # other is constant
                der_new = self.der
            else:  # self is constant
                N_new = other.N
                der_new = other.der
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val - other
                der_new = self.der
                N_new = self.N
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new, N_new)

    def __rsub__(self, other):
        # In this case, other must be a constant
        val_new = other - self.val
        der_new = -self.der
        return AutoDiff(val_new, der_new, self.N)

    def __mul__(self, other):
        # (f*g)' = f'*g + g' * f
        # (f*g)^(n) = sum_{k=0}^{n} [binom(n, k) * f^(n-k)g^(k)]
        try:
            val_new = self.val * other.val
            N_new = self.N
            if self.N == other.N:
                der_new = []
                for n in range(1, self.N+1):
                    # binomial(n, 0) = binomial(n, n) = 1 always holds
                    nth_der = self.der[-1] * other.val + self.val * other.der[-1]  # 1st and last term
                    for k in range(1, n):
                        nth_der += binomial(n, k) * self.der[n-k-1] * other.der[k-1]
                    der_new.append(nth_der)
            elif self.N > other.N:  # other is a constant
                der_new = self.der * other.val
            else:   # self is a constant
                der_new = other.der * self.val
                N_new = other.N

        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val * other
                der_new = self.der * other  # other is a constant in this case
                N_new = self.N
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new, N_new)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # (f/g)' = (f'*g - g'*f)/g^2
        try:
            if other.val != 0:
                # elementwise division to return Jacobian matrix
                denom = inv(other)
                return self.__mul__(denom)
            else:
                raise ZeroDivisionError('Division by zero')
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                if other != 0:
                    # just divide if other is constant
                    val_new = self.val / other
                    der_new = self.der / other
                    return AutoDiff(val_new, der_new, self.N)
                else:
                    raise ZeroDivisionError('Division by zero')
            else:
                raise AttributeError('Type error!')

    def __rtruediv__(self, other):
        if self.val != 0:
            inv_self = inv(self)
            return inv_self * other
        else:
            raise ZeroDivisionError('Division by zero')
        # val_new = other / self.val
        # der_new = - other * self.der/self.val**2
        # return AutoDiff(val_new, der_new)

    def __pow__(self, other):
        # (f^g)' = f^g * (f'/f * g + g' * ln(f))
        if self.val <= 0:
            raise ValueError('Error: Value of base function must be positive!')
        try:
            # elementwise power to return Jacobian matrix
            val_new = self.val ** other.val
            N_new = self.N  # The larger order of the two components
            if self.N == other.N:
                if N_new == 1:
                    der_new = val_new * (other.val * self.der / self.val + other.der * np.log(self.val))
                else:
                    # fx^gx = e^(gx * ln(fx))
                    pw = ln(self) * other
                    f_power_g = exp(pw)
                    return f_power_g
            elif self.N > other.N:
                # other is constant: x^a
                return power(self, other.val)
            else:
                # self is constant: b^gx = e^(gx * ln(b))
                pw = other * np.log(self.val)
                return exp(pw)
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                return power(self, other)
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new, N_new)

    def __rpow__(self, other):
        # other is constant: c^fx = e^(fx * ln(c))
        pw = self * np.log(other)
        return exp(pw)

    # unary operations
    def __neg__(self):
        val_new = -self.val
        der_new = -self.der
        return AutoDiff(val_new, der_new, self.N)

    def __pos__(self):
        val_new = self.val
        der_new = self.der
        return AutoDiff(val_new, der_new, self.N)

    # comparison operator
    def __lt__(self, other):
        """
        less than comparison operator

        Returns
        -------
        """
        try:
            return self.val < other.val
        except AttributeError:
            return self.val < other

    def __gt__(self, other):
        """
        greater than comparison operator
        Parameters
        ----------
        other

        Returns
        -------

        """
        try:
            return self.val > other.val
        except AttributeError:
            return self.val > other


    def __le__(self, other):
        """
        less than or equal to comparison operator
        Returns
        -------

        """
        try:
            return self.val <= other.val
        except AttributeError:
            return self.val <= other

    def __ge__(self, other):
        """
        greater than or equal to comparison operator
        Returns
        -------

        """
        try:
            return self.val >= other.val
        except AttributeError:
            return self.val >= other

    def __eq__(self, other):
        """
        equal to comparison operator
        Parameters
        ----------
        other

        Returns
        -------

        """
        try:
            return self.val == other.val
        except AttributeError:
            return self.val == other


    def __ne__(self, other):
        """
        not equal to comparison operator
        Returns
        -------

        """
        try:
            return not self.val == other.val
        except AttributeError:
            return not self.val == other

    def __str__(self):
        val = "val = " + str(self.val)
        der = "; der = " + str(self.der)
        return val + der



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

    if n == int(n) and n >= 0:
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
            der_new = n * x.val ** (n - 1) * x.der
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
    return AutoDiff(val_new, der_new, N)


def log(x, a):
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
    # equivalent to log_gx(fx) = ln(fx) / ln(gx)
    if isinstance(x, AutoDiff):
        if x.val <= 0:
            raise ValueError('Error: Independent variable must be positive!')
    try:
        return ln(x) / ln(a)
        # val_new = np.log(x.val) / np.log(n)
        # der_new = 1 / (x.val * np.log(n) * x.der)  # BUG: x.der should be outside???
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x <= 0:
                raise ValueError('Error: Independent variable must be positive!')
            return inv(ln(a)) * np.log(x)  # in case a is an AD object
        else:
            raise AttributeError('Type error!')

# def log_k_order(gx, n, k):
#     if k == 1:
#         return 1/(gx * np.log(n))
#     if k != 1:
#         return power_k_order(gx, -1, k-1)/(np.log(n))
#
# def log(x, n):
#     # (log_n(x))' = 1/(x * log_e(n) * x')
#     # we should also check the value >0 for log calculation
#     """Returns the value and derivative of a logarithm operation: log_n(x)
#
#     INPUTS
#     =======
#     x: an AutoDiff object or a scalar, required, the input variable
#     n: float or int, required, the base
#
#     RETURNS
#     ========
#     an AD object containing the value and derivative of the expression
#
#     EXAMPLES
#     =========
#     >>> log(np.e, np.e)
#     AD(1.0, 0)
#     >>> log(AD(np.e**2, 2.0), np.e)
#     AD(2.0, 0.06766764161830635)
#     """
#     if isinstance(x, AutoDiff):
#         if x.val <= 0:
#             raise ValueError('Error: Independent variable must be positive!')
#     N = 1
#     dk_f = lambda gx, k: log_k_order(gx, n, k)  # nth order derivative for log(x, n)
#     try:
#         val_new = np.log(x.val) / np.log(n)
#         N = x.N
#         if N == 1:
#             der_new = 1 / (x.val * np.log(n) * x.der)
#         else:
#             # N > 1
#             der_new = get_n_der_vecs(dk_f, x, N)
#     except AttributeError:
#         if isinstance(x, float) or isinstance(x, int):
#             if x <= 0:
#                 raise ValueError('Error: Independent variable must be positive!')
#             val_new = np.log(x) / np.log(n)
#             # If x is a constant, the derivative of x is 0.
#             der_new = np.zeros(1)
#         else:
#             raise AttributeError('Type error!')
#     return AutoDiff(val_new, der_new, N)


def ln(x):
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
    >>> ln(np.e**2)
    AD(2.0, 0)
    """
    if isinstance(x, AutoDiff):
        if x.val <= 0:
            raise ValueError('Error: Independent variable must be positive!')
    N = 1
    dk_f = lambda gx, k: ((-1.0)**(k-1) * factorial(k-1)) / (gx ** k)
    try:
        val_new = np.log(x.val)
        N = x.N
        if N == 1:
            der_new = x.der * (1 / x.val)
        else:
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x <= 0:
                raise ValueError('Error: Independent variable must be positive!')
            val_new = np.log(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AutoDiff(val_new, der_new, N)


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
            der_new = n**x.val * np.log(n) * x.der
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
    return AutoDiff(val_new, der_new, N)


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
            der_new = np.exp(x.val) * x.der
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
    return AutoDiff(val_new, der_new, N)


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
    return AutoDiff(val_new, der_new, N) # YW


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
    if isinstance(x, AutoDiff):
        if x.val < 0:
            raise ValueError('Error: Independent variable must be nonnegative!')
    N = 1
    dk_f = lambda gx, k: power_k_order(gx, 1/2, k)  # nth order derivative for sqrt(x)
    try:
        val_new = np.sqrt(x.val)
        N = x.N
        if N == 1:
            # der_new = np.array([1 / 2 * x.val ** (-1/2) * x.der])
            der_new = 0.5 * x.der / np.sqrt(x.val)
        else:
            # N > 1
            der_new = get_n_der_vecs(dk_f, x, N)
    except AttributeError:
        if isinstance(x, float) or isinstance(x, int):
            if x < 0:
                raise ValueError('Error: Independent variable must be nonnegative!')
            val_new = np.sqrt(x)
            # If x is a constant, the derivative of x is 0.
            der_new = np.zeros(1)
        else:
            raise AttributeError('Type error!')
    return AutoDiff(val_new, der_new, N)


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
    >>> sin(AutoDiff(0.0, 2.0))
    AD(0.0, 2.0)
    """
    half_pi = np.pi / 2
    N = 1
    dk_f = lambda gx, k: np.sin(gx + half_pi * k)  # nth order derivative for sin(x)
    try:
        val_new = np.sin(x.val)
        N = x.N
        if N == 1:
            der_new = np.cos(x.val) * x.der
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
    return AutoDiff(val_new, der_new, N)


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
            der_new = -np.sin(x.val) * x.der
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
    return AutoDiff(val_new, der_new, N)


def tan_k_order(gx, k):
    if k == 1:
        return 1 / (np.cos(gx)) ** 2
    else:
        x = AutoDiff(gx, N=k-1)
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
            der_new = np.array([1 / (np.cos(x.val)) ** 2])  # TODO: where is x.der?
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
    return AutoDiff(val_new, der_new, N)


def arcsin_k_order(gx, k):
    if k == 1:
        return 1 / np.sqrt(1 - gx**2)
    else:
        x = AutoDiff(gx, N=k-1)
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
    if isinstance(x, AutoDiff):
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
    return AutoDiff(val_new, der_new, N)


#### arccos is not correct for the 1th der when k>1
def arccos_k_order(gx, k):
    if k == 1:
        return - 1 / np.sqrt(1 - gx**2)
    else:
        x = AutoDiff(gx, N=k-1)
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
    if isinstance(x, AutoDiff):
        if x.val < -1 or x.val > 1:
            raise ValueError('Error: Independent variable must be in [-1,1]!')
    try:
        val_new = np.arccos(x.val)
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
    return AutoDiff(val_new, der_new, N)


def arctan_k_order(gx, k):
    if k == 1:
        return 1 / (1 + gx ** 2)
    else:
        x = AutoDiff(gx, N = k-1)
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
    return AutoDiff(val_new, der_new, N)


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
    return AutoDiff(val_new, der_new, N)


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
    return AutoDiff(val_new, der_new, N)


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
    >>> tanh(AutoDiff(0.0, N=2.0))
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
    return AutoDiff(val_new, der_new, N)
