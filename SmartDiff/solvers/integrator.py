import numpy as np

class AutoDiff():
    def __init__(self, value, der=1):
        self.val = value
        self.der = der

    def __add__(self, other):
        # (f+g)' = f' + g'
        try:
            # elementwise summation to return Jacobian matrix
            val_new = self.val + other.val
            der_new = self.der + other.der
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val + other
                der_new = self.der
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # (f-g)' = f' - g'
        try:
            # elementwise subtraction to return Jacobian matrix
            val_new = self.val - other.val
            der_new = self.der - other.der
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val - other
                der_new = self.der
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new)

    def __rsub__(self, other):
        val_new = other - self.val
        der_new = - self.der
        return AutoDiff(val_new, der_new)

    def __mul__(self, other):
        # (f*g)' = f'*g + g' * f
        try:
            # elementwise multiplication to return Jacobian matrix
            val_new = self.val * other.val
            der_new = self.der * other.val + other.der * self.val
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val * other
                der_new = self.der * other
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # (f/g)' = (f'*g - g'*f)/g^2
        try:
            if other.val != 0:
                # elementwise division to return Jacobian matrix
                val_new = self.val / other.val
                der_new = (self.der * other.val -  other.der * self.val) / (other.val)**2
            else:
                raise ZeroDivisionError('Division by zero')
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                if other != 0:
                    val_new = self.val / other
                    der_new = self.der / other
                else:
                    raise ZeroDivisionError('Division by zero')
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new)

    def __rtruediv__(self, other):
        val_new = other/self.val
        der_new = - other * self.der/self.val**2
        return AutoDiff(val_new, der_new)

    def __pow__(self, other):
        # (f^g)' = f^g * (f'/f * g + g' * ln(f))
        if self.val <= 0:
            raise ValueError('Error: Value of base function must be positive!')
        try:
            # elementwise power to return Jacobian matrix
            val_new = self.val ** other.val
            der_new = self.val ** other.val * (self.der/self.val * other.val + other.der * np.log(self.val))
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = self.val ** other
                der_new = self.val ** other * self.der/self.val * other
            else:
                raise AttributeError('Type error!')
        return AutoDiff(val_new, der_new)

    def __rpow__(self, other):
        val_new = other ** self.val
        der_new = other ** self.val * self.der * np.log(other)
        return AutoDiff(val_new, der_new)

    # unary operations
    def __neg__(self):
        val_new = - self.val
        der_new = - self.der
        return AutoDiff(val_new, der_new)

    def __pos__(self):
        val_new = self.val
        der_new = self.der
        return AutoDiff(val_new, der_new)
        

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
            return self.val > val


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