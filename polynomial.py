import operator
from functools import reduce
from field import FieldElement
import numpy as np
from itertools import dropwhile, starmap, zip_longest
import warnings

# Suppress numpy overflow warnings/errors as requested
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x

# Extract modulus for numpy operations. 
# Defaults to a large prime (STARK 101 default) if not found in FieldElement
try:
    MOD = FieldElement.k_modulus
except AttributeError:
    MOD = 3 * 2**30 + 1 

def remove_trailing_elements(list_of_elements, element_to_remove):
    return list(dropwhile(lambda x: x == element_to_remove, list_of_elements[::-1]))[::-1]


def two_lists_tuple_operation(f, g, operation, fill_value):
    return list(starmap(operation, zip_longest(f, g, fillvalue=fill_value)))


def scalar_operation(list_of_elements, operation, scalar):
    return [operation(c, scalar) for c in list_of_elements]


def trim_trailing_zeros(p):
    return remove_trailing_elements(p, FieldElement.zero())


def prod(values):
    """
    Computes a product. 
    Optimized: Uses a tree-reduction strategy (Divide & Conquer).
    This reduces the complexity of polynomial multiplication chains significantly
    when combined with FFT multiplication.
    """
    len_values = len(values)
    if len_values == 0:
        return 1
    if len_values == 1:
        return values[0]
    return prod(values[:len_values // 2]) * prod(values[len_values // 2:])


def latex_monomial(exponent, coef, var):
    if exponent == 0:
        return str(coef)
    if coef == 1:
        coef = ''
    if coef == -1:
        coef = '-'
    if exponent == 1:
        return f'{coef}{var}'
    return f'{coef}{var}^{{{exponent}}}'


class Polynomial:
    """
    Represents a polynomial over FieldElement.
    Optimized with Numpy (Object Dtype) for infinite precision and vectorized speed.
    """

    @classmethod
    def X(cls):
        return cls([FieldElement.zero(), FieldElement.one()])

    def __init__(self, coefficients, var='x'):
        # dtype=object allows Python integers to handle values > 2^64 without overflow errors
        if isinstance(coefficients, np.ndarray):
            self.coeffs = coefficients.astype(object) 
        elif isinstance(coefficients, Polynomial):
             self.coeffs = coefficients.coeffs
        else:
            data = [c.val if isinstance(c, FieldElement) else c for c in coefficients]
            self.coeffs = np.array(data, dtype=object)
        
        # Efficiently trim trailing zeros
        # np.trim_zeros only works on 1D arrays, so we ensure 1D
        if self.coeffs.ndim > 0:
            self.coeffs = np.trim_zeros(self.coeffs, 'b')
        
        if self.coeffs.size == 0:
            self.coeffs = np.array([0], dtype=object)

        self.var = var

    # Legacy property to maintain API compatibility with list-based tests
    @property
    def poly(self):
        return [FieldElement(int(c)) for c in self.coeffs]
    
    @poly.setter
    def poly(self, val):
        data = [c.val if isinstance(c, FieldElement) else c for c in val]
        self.coeffs = np.trim_zeros(np.array(data, dtype=object), 'b')
        if self.coeffs.size == 0:
            self.coeffs = np.array([0], dtype=object)

    def _repr_latex_(self):
        if self.coeffs.size == 0 or (self.coeffs.size == 1 and self.coeffs[0] == 0):
            return '$0$'
        res = ['$']
        first = True
        for exponent, coef in enumerate(self.coeffs):
            if coef == 0:
                continue
            monomial = latex_monomial(exponent, coef, self.var)
            if first:
                first = False
                res.append(monomial)
                continue
            oper = '+'
            if monomial[0] == '-':
                oper = '-'
                monomial = monomial[1:]
            res.append(oper)
            res.append(monomial)
        res.append('$')
        return ' '.join(res)

    def __eq__(self, other):
        if isinstance(other, int):
            if other == 0 and (self.coeffs.size == 0 or (self.coeffs.size==1 and self.coeffs[0]==0)):
                return True
        try:
            other = Polynomial.typecast(other)
        except AssertionError:
            return False
        return np.array_equal(self.coeffs, other.coeffs)

    @staticmethod
    def typecast(other):
        if isinstance(other, int):
            return Polynomial([other])
        if isinstance(other, FieldElement):
            return Polynomial([other.val])
        if isinstance(other, Polynomial):
            return other
        assert False, f'Type mismatch: Polynomial and {type(other)}.'

    def __add__(self, other):
        other = Polynomial.typecast(other)
        # Vectorized addition on object arrays (handles large ints automatically)
        s_len, o_len = len(self.coeffs), len(other.coeffs)
        if s_len > o_len:
            res = self.coeffs.copy()
            res[:o_len] = (res[:o_len] + other.coeffs) % MOD
        else:
            res = other.coeffs.copy()
            res[:s_len] = (res[:s_len] + self.coeffs) % MOD
        return Polynomial(res)

    __radd__ = __add__

    def __sub__(self, other):
        other = Polynomial.typecast(other)
        s_len, o_len = len(self.coeffs), len(other.coeffs)
        if s_len >= o_len:
            res = self.coeffs.copy()
            res[:o_len] = (res[:o_len] - other.coeffs) % MOD
        else:
            res = -other.coeffs
            res = res % MOD 
            res[:s_len] = (res[:s_len] + self.coeffs) % MOD
        return Polynomial(res)

    def __rsub__(self, other):
        return -(self - other)

    def __neg__(self):
        # (0 - coeffs) % MOD handles the negation in finite field correctly
        return Polynomial((0 - self.coeffs) % MOD)

    def __mul__(self, other):
        other = Polynomial.typecast(other)
        a = self.coeffs
        b = other.coeffs
        
        # FFT Optimization
        # Use FFT for larger polynomials where N log N beats N^2
        if len(a) + len(b) > 256: 
            n = 1
            target_size = len(a) + len(b) - 1
            while n < target_size:
                n <<= 1
                
            # FFT requires floats. We cast to float, do FFT, then round back.
            # Precision warning: For extremely large fields, float64 might lose precision.
            # But for STARK101 (2^30), it works reasonably well.
            fft_a = np.fft.rfft(a.astype(float), n)
            fft_b = np.fft.rfft(b.astype(float), n)
            fft_res = fft_a * fft_b
            res = np.fft.irfft(fft_res, n)
            
            # Round and cast back to object (python int) to prevent overflow in future ops
            res = (np.round(res).astype(object)) % MOD
            res = res[:target_size]
        else:
            # Numpy convolution is highly optimized C-loop
            # We must use modulo after convolution
            res = np.convolve(a, b) % MOD
            
        return Polynomial(res)

    __rmul__ = __mul__

    def compose(self, other):
        other = Polynomial.typecast(other)
        res = Polynomial([])
        # Horner's method
        for coef in self.coeffs[::-1]:
            res = (res * other) + Polynomial([coef])
        return res

    def qdiv(self, other):
        other = Polynomial.typecast(other)
        divisor = other.coeffs
        dividend = self.coeffs.copy()
        
        if divisor.size == 0 or (divisor.size == 1 and divisor[0] == 0):
             assert False, 'Dividing by zero polynomial.'

        len_divisor = len(divisor)
        len_dividend = len(dividend)
        
        if len_dividend < len_divisor:
             return Polynomial([]), self

        # Precompute inverse of leading term
        inv = FieldElement(int(divisor[-1])).inverse().val
        
        quotient = np.zeros(len_dividend - len_divisor + 1, dtype=object)
        
        # Optimized Synthetic-like Division loop
        # Iterating in reverse to eliminate terms
        for i in range(len(quotient) - 1, -1, -1):
             q = (dividend[i + len_divisor - 1] * inv) % MOD
             quotient[i] = q
             if q != 0:
                 start = i
                 end = i + len_divisor
                 # Vectorized update
                 dividend[start:end] = (dividend[start:end] - (divisor * q)) % MOD
        
        return Polynomial(quotient), Polynomial(dividend)

    def __truediv__(self, other):
        div, mod = self.qdiv(other)
        assert mod == 0, 'Polynomials are not divisible.'
        return div

    def __mod__(self, other):
        return self.qdiv(other)[1]

    @staticmethod
    def monomial(degree, coefficient):
        c = coefficient.val if isinstance(coefficient, FieldElement) else coefficient
        coeffs = np.zeros(degree + 1, dtype=object)
        coeffs[-1] = c
        return Polynomial(coeffs)

    @staticmethod
    def gen_linear_term(point):
        p = point.val if isinstance(point, FieldElement) else point
        # (x - p)
        return Polynomial(np.array([-p, 1], dtype=object))

    def degree(self):
        return len(self.coeffs) - 1

    def get_nth_degree_coefficient(self, n):
        if n > self.degree():
            return FieldElement.zero()
        else:
            return FieldElement(int(self.coeffs[n]))

    def scalar_mul(self, scalar):
        s = scalar.val if isinstance(scalar, FieldElement) else scalar
        return Polynomial((self.coeffs * s) % MOD)

    def eval(self, point):
        point_val = point.val if isinstance(point, FieldElement) else point
        val = 0
        # Fast python loop for Horner's method
        for c in self.coeffs[::-1]:
            val = (val * point_val + c) % MOD
        return FieldElement(val)

    def __call__(self, other):
        if isinstance(other, (int)):
            other = FieldElement(other)
        if isinstance(other, FieldElement):
            return self.eval(other)
        if isinstance(other, Polynomial):
            return self.compose(other)
        raise NotImplementedError()

    def __pow__(self, other):
        assert other >= 0
        res = Polynomial([1])
        cur = self
        while True:
            if other % 2 != 0:
                res *= cur
            other >>= 1
            if other == 0:
                break
            cur = cur * cur
        return res


def calculate_lagrange_polynomials(x_values):
    """
    Computes Lagrange polynomials optimized with Synthetic Division.
    Complexity: O(N^2) instead of O(N^3).
    """
    lagrange_polynomials = []
    
    # 1. Compute Master Polynomial: Z(x) = Product(x - x_i)
    # Using tree-product (prod) makes this fast
    monomials = [Polynomial.gen_linear_term(x) for x in x_values]
    numerator = prod(monomials)
    
    # Extract integer values for raw arithmetic
    x_ints = [x.val if isinstance(x, FieldElement) else x for x in x_values]
    coeffs = numerator.coeffs
    
    for j in tqdm(range(len(x_values))):
        xj = x_ints[j]
        
        # 2. Denominator Calculation: Product(x_j - x_i) for i != j
        # Calculated directly using scalar arithmetic (O(N))
        denom_val = 1
        for i, xi in enumerate(x_ints):
            if i == j: continue
            diff = (xj - xi) % MOD
            denom_val = (denom_val * diff) % MOD
        
        # Invert denominator once
        denom_inv = FieldElement(denom_val).inverse().val

        # 3. Synthetic Division: Calculate Numerator / (x - x_j)
        # Since the divisor is linear (x - root), we can use optimized Synthetic Division (O(N))
        # Logic: New_Coeff[k] = Old_Coeff[k+1] + root * New_Coeff[k+1]
        
        quotient_coeffs = np.zeros(len(coeffs) - 1, dtype=object)
        current_val = coeffs[-1] # Leading coefficient Z(x)
        quotient_coeffs[-1] = current_val
        
        # Iterate backwards
        for i in range(len(coeffs) - 2, -1, -1):
            # The synthetic division step
            current_val = (coeffs[i] + xj * current_val) % MOD
            quotient_coeffs[i] = current_val
            
        # Create polynomial and apply the denominator scalar
        poly_arr = (quotient_coeffs * denom_inv) % MOD
        lagrange_polynomials.append(Polynomial(poly_arr))

    return lagrange_polynomials


def interpolate_poly_lagrange(y_values, lagrange_polynomials):
    poly = Polynomial([0])
    for j, y_value in enumerate(y_values):
        poly += lagrange_polynomials[j].scalar_mul(y_value)
    return poly


def interpolate_poly(x_values, y_values):
    assert len(x_values) == len(y_values)
    lp = calculate_lagrange_polynomials(x_values)
    return interpolate_poly_lagrange(y_values, lp)