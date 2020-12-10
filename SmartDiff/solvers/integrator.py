import numpy as np
from sympy import binomial
import SmartDiff.solvers.element_op as el


## The reason I change the type to array and then back list is to make sure it can do elementwise computation to save Jacobian matrix for multiple functions.
## In stead of doing for loop, I guess it will save more time.

class AutoDiff():
    def __init__(self, value, der=None, N=1):
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
                N_new = other.der

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
                denom = el.inv(other)
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
            inv_self = el.inv(self)
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
                    der_new = val_new * (other.val * self.der / self.val + other.der / np.log(self.val))
                else:
                    # fx^gx = e^(gx * ln(fx))
                    pw = el.ln(self) * other
                    f_power_g = el.exp(pw)
                    return f_power_g
            elif self.N > other.N:
                # other is constant: x^a
                return el.power(self, other.val)
            else:
                # self is constant: b^gx = e^(gx * ln(b))
                pw = other * np.log(self.val)
                return el.exp(pw)
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                return el.power(self, other)
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new, N_new)

    def __rpow__(self, other):
        # other is constant: c^fx
        pw = self.val * np.log(other)
        return el.exp(pw)

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

