import numpy as np

## The reason I change the type to array and then back list is to make sure it can do elementwise computation to save Jacobian matrix for multiple functions.
## In stead of doing for loop, I guess it will save more time.

class AutoDiffToy():
    def __init__(self, value, der=1):
        self.val = value
        self.der = der

    def __add__(self, other):
        # (f+g)' = f' + g'
        try:
            # elementwise summation to return Jacobian matrix
            val_new = (np.array(self.val) + np.array(other.val)).tolist()
            der_new = (np.array(self.der) + np.array(other.der)).tolist()
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = (np.array(self.val) + np.array(other)).tolist()
                der_new = self.der
            else:
                raise AttributeError('Type error!')
        return AutoDiffToy(val_new, der_new)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # (f-g)' = f' - g'
        try:
            # elementwise subtraction to return Jacobian matrix
            val_new = (np.array(self.val) - np.array(other.val)).tolist()
            der_new = (np.array(self.der) - np.array(other.der)).tolist()
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = (np.array(self.val) - np.array(other)).tolist()
                der_new = self.der
            else:
                raise AttributeError('Type error!')
        return AutoDiffToy(val_new, der_new)

    def __rsub__(self, other):
        val_new = (np.array(other) - np.array(self.val)).tolist()
        der_new = - self.der
        return AutoDiffToy(val_new, der_new)

    def __mul__(self, other):
        # (f*g)' = f'*g + g' * f
        try:
            # elementwise multiplication to return Jacobian matrix
            val_new = (np.array(self.val) * np.array(other.val)).tolist()
            der_new = (np.array(self.der) * np.array(other.val) + np.array(other.der) * np.array(self.val)).tolist()
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = (np.array(self.val) * np.array(other)).tolist()
                der_new = (np.array(self.der) * np.array(other)).tolist()
            else:
                raise AttributeError('Type error!')
        return AutoDiffToy(val_new, der_new)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # (f/g)' = (f'*g - g'*f)/g^2
        try:
            # elementwise division to return Jacobian matrix
            val_new = (np.array(self.val) / np.array(other.val)).tolist()
            der_new = ((np.array(self.der) * np.array(other.val) -  np.array(other.der) * np.array(self.val)) / np.array(other.val)**2).tolist()
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                if other != 0:
                    val_new = (np.array(self.val) / np.array(other)).tolist()
                    der_new = (np.array(self.der) / np.array(other)).tolist()
                else:
                    raise ZeroDivisionError('Division by zero')
            else:
                raise AttributeError('Type error!')
        return AutoDiffToy(val_new, der_new)

    def __rtruediv__(self, other):
        val_new = other/self.val
        der_new = - other * self.der/self.val**2
        return AutoDiffToy(val_new, der_new)

    def __pow__(self, other):
        # (f^g)' = f^g * (f'/f * g + g' * ln(f))
        if self.val <= 0:
            raise ValueError('Error: Value of base function must be positive!')
        try:
            # elementwise power to return Jacobian matrix
            val_new = (np.array(self.val) ** np.array(other.val)).tolist()
            der_new = (np.array(self.val) ** np.array(other.val) * (np.array(self.der)/np.array(self.val) * np.array(other.val) + np.array(other.der) * np.array(np.log(self.val)))).tolist()
        except AttributeError:
            if isinstance(other, float) or isinstance(other, int):
                val_new = (np.array(self.val) ** np.array(other)).tolist()
                der_new = (np.array(self.val) ** np.array(other) * (np.array(self.der)/np.array(self.val) * np.array(other))).tolist()
            else:
                raise AttributeError('Type error!')
        return AutoDiffToy(val_new, der_new)

    def __rpow__(self, other):
        val_new = (np.array(other) ** np.array(self.val)).tolist()
        der_new = (np.array(other) ** np.array(self.val) * (np.array(self.der) * np.array(np.log(other)))).tolist()
        return AutoDiffToy(val_new, der_new)

    # unary operations
    def __neg__(self):
        val_new = - self.val
        der_new = - self.der
        return AutoDiffToy(val_new, der_new)

    def __pos__(self):
        val_new = self.val
        der_new = self.der
        return AutoDiffToy(val_new, der_new)
