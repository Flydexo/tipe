from random import randint

class FieldElement:
    """
    Represents an element of F_(3 * 2**30 + 1).
    """
    k_modulus = 3 * 2**30 + 1
    generator_val = 5

    def __init__(self, val):
        # FIX: Force conversion to Python int to prevent Numpy int64 overflow
        self.val = int(val) % FieldElement.k_modulus

    @staticmethod
    def zero():
        return FieldElement(0)

    @staticmethod
    def one():
        return FieldElement(1)

    def __repr__(self):
        return repr((self.val + self.k_modulus//2) % self.k_modulus - self.k_modulus//2)

    def __eq__(self, other):
        if isinstance(other, int):
            other = FieldElement(other)
        return isinstance(other, FieldElement) and self.val == other.val

    def __hash__(self):
        return hash(self.val)

    @staticmethod
    def generator():
        return FieldElement(FieldElement.generator_val)

    @staticmethod
    def typecast(other):
        # FIX: Check for generic integer types to handle numpy.int64 gracefully
        if hasattr(other, 'dtype') or isinstance(other, int) or isinstance(other, float):
             return FieldElement(int(other))
        assert isinstance(other, FieldElement), f'Type mismatch: FieldElement and {type(other)}.'
        return other

    def __neg__(self):
        return self.zero() - self

    def __add__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        return FieldElement(self.val + other.val)

    __radd__ = __add__

    def __sub__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        return FieldElement(self.val - other.val)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        try:
            other = FieldElement.typecast(other)
        except AssertionError:
            return NotImplemented
        # The overflow happened here because self.val was np.int64
        return FieldElement(self.val * other.val)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = FieldElement.typecast(other)
        return self * other.inverse()

    def __pow__(self, n):
        assert n >= 0
        cur_pow = self
        res = FieldElement(1)
        while n > 0:
            if n % 2 != 0:
                res *= cur_pow
            n = n // 2
            cur_pow *= cur_pow
        return res

    def inverse(self):
        t, new_t = 0, 1
        r, new_r = FieldElement.k_modulus, self.val
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, (t - (quotient * new_t))
            r, new_r = new_r, r - quotient * new_r
        assert r == 1
        return FieldElement(t)

    def is_order(self, n):
        assert n >= 1
        h = FieldElement(1)
        for _ in range(1, n):
            h *= self
            if h == FieldElement(1):
                return False
        return h * self == FieldElement(1)

    def _serialize_(self):
        return repr(self.val)

    @staticmethod
    def random_element(exclude_elements=[]):
        fe = FieldElement(randint(0, FieldElement.k_modulus - 1))
        while fe in exclude_elements:
            fe = FieldElement(randint(0, FieldElement.k_modulus - 1))
        return fe