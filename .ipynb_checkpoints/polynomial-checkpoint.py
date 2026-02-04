import operator
from functools import reduce
from field import FieldElement
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x

from itertools import dropwhile, starmap, zip_longest

# --- FFT/NTT Helpers ---

def _extended_gcd(a, b):
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t

def _get_modular_inverse(n, p):
    gcd, x, y = _extended_gcd(n, p)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {n} mod {p}")
    return x % p

def _get_primitive_root(p):
    # Simple check for small primes or standard hardcoded ones could go here.
    # We use a randomized search approach which is efficient.
    if p == 2: return 1
    if p == 998244353: return 3  # Common NTT prime
    
    # Factor p-1
    n = p - 1
    factors = set()
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
        
    # Check generators
    for g in range(2, p):
        if all(pow(g, n // f, p) != 1 for f in factors):
            return g
    return None

class _NTTContext:
    """Singleton to cache roots of unity for the field."""
    _instance = None
    
    def __new__(cls, modulus):
        if cls._instance is None or cls._instance.modulus != modulus:
            cls._instance = super(_NTTContext, cls).__new__(cls)
            cls._instance.modulus = modulus
            cls._instance.roots = {} # cache by domain size
            cls._instance.inv_roots = {}
            
            # Check 2-adicity
            s = 0
            t = modulus - 1
            while t % 2 == 0:
                t //= 2
                s += 1
            cls._instance.s = s
            cls._instance.t = t
            cls._instance.root_of_unity = _get_primitive_root(modulus)
        return cls._instance

    def get_roots(self, n):
        if n in self.roots:
            return self.roots[n], self.inv_roots[n]
        
        # n must be a power of 2
        k = n.bit_length() - 1
        if k > self.s:
            return None, None # Modulus doesn't support this size NTT
            
        g = pow(self.root_of_unity, (self.modulus - 1) // n, self.modulus)
        inv_g = _get_modular_inverse(g, self.modulus)
        
        # Precompute bit-reversed roots
        roots = [0] * n
        inv_roots = [0] * n
        roots[0] = 1
        inv_roots[0] = 1
        
        for i in range(1, n):
            roots[i] = (roots[i-1] * g) % self.modulus
            inv_roots[i] = (inv_roots[i-1] * inv_g) % self.modulus

        # Bit reversal permutation
        rev_roots = [0] * n
        rev_inv_roots = [0] * n
        for i in range(n):
            # Compute bit reverse of i
            rev = 0
            temp = i
            for _ in range(k):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            rev_roots[rev] = roots[i]
            rev_inv_roots[rev] = inv_roots[i]

        self.roots[n] = rev_roots
        self.inv_roots[n] = rev_inv_roots
        return rev_roots, rev_inv_roots

def _ntt(coeffs, inverse=False):
    n = len(coeffs)
    modulus = FieldElement.k_modulus
    ctx = _NTTContext(modulus)
    roots, inv_roots = ctx.get_roots(n)
    
    if roots is None:
        raise ValueError("Field does not support NTT of this size")
        
    w_table = inv_roots if inverse else roots
    
    # Bit-reverse copy (iterative NTT)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            coeffs[i], coeffs[j] = coeffs[j], coeffs[i]

    # Butterfly operations
    length = 2
    while length <= n:
        half_len = length // 2
        w_step = n // length
        for i in range(0, n, length):
            w_idx = 0
            for k in range(half_len):
                # The w_table is already bit-reversed, logic here simplifies to standard
                # Cooley-Tukey if we were using standard roots. 
                # For pre-computed bit-reversed roots, we access them differently or
                # simply recompute standard butterfly factors. 
                # To be "blazingly fast" and safe without complex indexing:
                # We use standard iterative structure with precomputed powers.
                
                # Re-computing factor for clarity & correctness in this specific snippet:
                # (Optimization: In a full C++ engine we'd use the precomputed array better)
                # Here we just grab the correct root from the table for the block.
                
                # Standard Cooley-Tukey access pattern:
                # w = w_n^k
                # current_w = pow(g, (modulus-1)/length * k)
                # But we want to use the table.
                
                # Simple iterative fallback for the butterfly to ensure correctness:
                # This is the "internal" loop
                pass
        length <<= 1
    
    # Re-implementing a simpler iterative NTT to ensure no indexing bugs with the table above
    # Using standard Cooley-Tukey with precomputed roots
    
    # 1. Standard Bit Reversal
    # (Already done above)
    
    # 2. Transform
    m = 1
    while m < n:
        # w_m is primitive 2m-th root of unity
        # w_m = g ^ ((P-1)/2m)
        base_w = pow(ctx.root_of_unity, (modulus - 1) // (2 * m), modulus)
        if inverse:
            base_w = _get_modular_inverse(base_w, modulus)
            
        w = 1
        for j in range(m):
            for i in range(j, n, 2 * m):
                t = (w * coeffs[i + m]) % modulus
                u = coeffs[i]
                coeffs[i] = (u + t) % modulus
                coeffs[i + m] = (u - t) % modulus
            w = (w * base_w) % modulus
        m *= 2

    if inverse:
        inv_n = _get_modular_inverse(n, modulus)
        for i in range(n):
            coeffs[i] = (coeffs[i] * inv_n) % modulus
            
    return coeffs

# --- End FFT Helpers ---


def remove_trailing_elements(list_of_elements, element_to_remove):
    return list(dropwhile(lambda x: x == element_to_remove, list_of_elements[::-1]))[::-1]


def two_lists_tuple_operation(f, g, operation, fill_value):
    return list(starmap(operation, zip_longest(f, g, fillvalue=fill_value)))


def scalar_operation(list_of_elements, operation, scalar):
    return [operation(c, scalar) for c in list_of_elements]


def trim_trailing_zeros(p):
    """
    Removes zeros from the end of a list.
    """
    return remove_trailing_elements(p, FieldElement.zero())


def prod(values):
    """
    Computes a product.
    """
    len_values = len(values)
    if len_values == 0:
        return 1
    if len_values == 1:
        return values[0]
    return prod(values[:len_values // 2]) * prod(values[len_values // 2:])


def latex_monomial(exponent, coef, var):
    """
    Returns a string representation of a monomial as LaTeX.
    """
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
    """

    @classmethod
    def X(cls):
        """
        Returns the polynomial x.
        """
        return cls([FieldElement.zero(), FieldElement.one()])

    def __init__(self, coefficients, var='x'):
        # Internally storing the coefficients in self.poly, least-significant (i.e. free term)
        # first, so $9 - 3x^2 + 19x^5$ is represented internally by the list  [9, 0, -3, 0, 0, 19].
        # Note that coefficients is copied, so the caller may freely modify the given argument.
        self.poly = remove_trailing_elements(coefficients, FieldElement.zero())
        self.var = var

    def _repr_latex_(self):
        """
        Returns a LaTeX representation of the Polynomial, for Jupyter.
        """
        if not self.poly:
            return '$0$'
        res = ['$']
        first = True
        for exponent, coef in enumerate(self.poly):
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
        try:
            other = Polynomial.typecast(other)
        except AssertionError:
            return False
        return self.poly == other.poly

    @staticmethod
    def typecast(other):
        """
        Constructs a Polynomial from `FieldElement` or `int`.
        """
        if isinstance(other, int):
            other = FieldElement(other)
        if isinstance(other, FieldElement):
            other = Polynomial([other])
        assert isinstance(other, Polynomial), f'Type mismatch: Polynomial and {type(other)}.'
        return other

    def __add__(self, other):
        other = Polynomial.typecast(other)
        return Polynomial(two_lists_tuple_operation(
            self.poly, other.poly, operator.add, FieldElement.zero()))

    __radd__ = __add__  # To support <int> + <Polynomial> (as in `1 + x + x**2`).

    def __sub__(self, other):
        other = Polynomial.typecast(other)
        return Polynomial(two_lists_tuple_operation(
            self.poly, other.poly, operator.sub, FieldElement.zero()))

    def __rsub__(self, other):  # To support <int> - <Polynomial> (as in `1 - x + x**2`).
        return -(self - other)

    def __neg__(self):
        return Polynomial([]) - self

    def __mul__(self, other):
        other = Polynomial.typecast(other)
        
        # Optimization: Early exit for zero polynomials
        if not self.poly or not other.poly:
            return Polynomial([])

        deg1 = self.degree()
        deg2 = other.degree()
        target_deg = deg1 + deg2
        
        # Optimization: Use schoolbook multiplication for small polynomials
        # Overhead of NTT is only worth it for N > ~64
        if deg1 < 64 or deg2 < 64:
            pol1, pol2 = [[x.val for x in p.poly] for p in (self, other)]
            res = [0] * (target_deg + 1)
            modulus = FieldElement.k_modulus
            for i, c1 in enumerate(pol1):
                if c1 == 0: continue
                for j, c2 in enumerate(pol2):
                    res[i + j] = (res[i + j] + c1 * c2) % modulus
            return Polynomial([FieldElement(x) for x in res])

        # NTT Multiplication
        try:
            # 1. Find size N as power of 2
            n = 1
            while n <= target_deg:
                n <<= 1
                
            # 2. Extract integers (avoid object overhead)
            poly_vals_a = [x.val for x in self.poly] + [0] * (n - len(self.poly))
            poly_vals_b = [x.val for x in other.poly] + [0] * (n - len(other.poly))
            
            # 3. NTT
            # If the field modulus is not NTT-friendly, this might raise or return garbage 
            # if we didn't check. The helper checks availability.
            ntt_a = _ntt(poly_vals_a, inverse=False)
            ntt_b = _ntt(poly_vals_b, inverse=False)
            
            # 4. Pointwise Multiplication
            modulus = FieldElement.k_modulus
            ntt_res = [(a * b) % modulus for a, b in zip(ntt_a, ntt_b)]
            
            # 5. Inverse NTT
            res_vals = _ntt(ntt_res, inverse=True)
            
            # 6. Trim to expected degree (handles precision noise if float FFT was used, 
            # but here we use integer NTT so it's exact, just trimming trailing zeros)
            # and wrap back to FieldElements
            return Polynomial([FieldElement(x) for x in res_vals[:target_deg+1]])
            
        except ValueError:
            # Fallback to O(N^2) if field is not NTT friendly
            pol1, pol2 = [[x.val for x in p.poly] for p in (self, other)]
            res = [0] * (target_deg + 1)
            modulus = FieldElement.k_modulus
            for i, c1 in enumerate(pol1):
                for j, c2 in enumerate(pol2):
                    res[i + j] = (res[i + j] + c1 * c2) % modulus
            return Polynomial([FieldElement(x) for x in res])

    __rmul__ = __mul__  # To support <int> * <Polynomial>.

    def compose(self, other):
        """
        Composes this polynomial with `other`.
        Example:
        >>> f = X**2 + X
        >>> g = X + 1
        >>> f.compose(g) == (2 + 3*X + X**2)
        True
        """
        other = Polynomial.typecast(other)
        res = Polynomial([])
        # Horner's method for composition
        for coef in self.poly[::-1]:
            res = (res * other) + Polynomial([coef])
        return res

    def qdiv(self, other):
        """
        Returns q, r the quotient and remainder polynomials respectively, such that
        f = q * g + r, where deg(r) < deg(g).
        * Assert that g is not the zero polynomial.
        """
        other = Polynomial.typecast(other)
        pol2 = trim_trailing_zeros(other.poly)
        assert pol2, 'Dividing by zero polynomial.'
        pol1 = trim_trailing_zeros(self.poly)
        if not pol1:
            return Polynomial([]), Polynomial([])
            
        # Optimization: Synthetic Division (Ruffini's Rule) for linear divisors
        # If dividing by (x - c), which is [ -c, 1 ]
        # This speeds up Lagrange interpolation massively (O(N) vs O(N^2))
        if len(pol2) == 2 and pol2[1] == FieldElement.one():
            c = -pol2[0] # divisor is x - c
            c_val = c.val
            modulus = FieldElement.k_modulus
            
            # Working with integers for speed
            coeffs = [x.val for x in pol1]
            deg = len(coeffs) - 1
            quotient_vals = [0] * deg
            
            remainder_val = 0
            # Standard Synthetic Division
            for i in range(deg, 0, -1):
                # coeff of x^i in numerator becomes coeff of x^(i-1) in quotient
                # But we must add the carry from the previous step
                # A better loop for synthetic division of P(x) / (x-c):
                # Q[i] = P[i+1] + c * Q[i+1] (iterating downwards)
                pass 
            
            # Let's do it cleanly:
            # P(x) = a_n x^n + ... + a_0
            # Q(x) = b_{n-1} x^{n-1} + ... + b_0
            # b_{n-1} = a_n
            # b_{k-1} = a_k + c * b_k
            
            if deg < 0: return Polynomial([]), Polynomial(pol1)

            b = coeffs[deg] # Leading coeff
            quotient_vals[deg-1] = b
            
            for k in range(deg-1, 0, -1):
                b = (coeffs[k] + c_val * b) % modulus
                quotient_vals[k-1] = b
            
            remainder_val = (coeffs[0] + c_val * b) % modulus
            
            return Polynomial([FieldElement(x) for x in quotient_vals]), Polynomial([FieldElement(remainder_val)])

        # Standard Long Division O(N^2)
        rem = pol1
        deg_dif = len(rem) - len(pol2)
        quotient = [FieldElement.zero()] * (deg_dif + 1)
        g_msc_inv = pol2[-1].inverse()
        while deg_dif >= 0:
            tmp = rem[-1] * g_msc_inv
            quotient[deg_dif] = quotient[deg_dif] + tmp
            last_non_zero = deg_dif - 1
            for i, coef in enumerate(pol2, deg_dif):
                rem[i] = rem[i] - (tmp * coef)
                if rem[i] != FieldElement.zero():
                    last_non_zero = i
            # Eliminate trailing zeroes (i.e. make r end with its last non-zero coefficient).
            rem = rem[:last_non_zero + 1]
            deg_dif = len(rem) - len(pol2)
        return Polynomial(trim_trailing_zeros(quotient)), Polynomial(rem)

    def __truediv__(self, other):
        div, mod = self.qdiv(other)
        assert mod == 0, 'Polynomials are not divisible.'
        return div

    def __mod__(self, other):
        return self.qdiv(other)[1]

    @staticmethod
    def monomial(degree, coefficient):
        """
        Constructs the monomial coefficient * x**degree.
        """
        return Polynomial([FieldElement.zero()] * degree + [coefficient])

    @staticmethod
    def gen_linear_term(point):
        """
        Generates the polynomial (x-p) for a given point p.
        """
        return Polynomial([FieldElement.zero() - point, FieldElement.one()])

    def degree(self):
        """
        The polynomials are represented by a list so the degree is the length of the list minus the
        number of trailing zeros (if they exist) minus 1.
        This implies that the degree of the zero polynomial will be -1.
        """
        return len(trim_trailing_zeros(self.poly)) - 1

    def get_nth_degree_coefficient(self, n):
        """
        Returns the coefficient of x**n
        """
        if n > self.degree():
            return FieldElement.zero()
        else:
            return self.poly[n]

    def scalar_mul(self, scalar):
        """
        Multiplies polynomial by a scalar
        """
        if scalar == FieldElement.zero():
            return Polynomial([])
        return Polynomial(scalar_operation(self.poly, operator.mul, scalar))

    def eval(self, point):
        """
        Evaluates the polynomial at the given point using Horner evaluation.
        """
        point = FieldElement.typecast(point).val
        # Doing this with ints (as opposed to `FieldElement`s) speeds up eval significantly.
        val = 0
        modulus = FieldElement.k_modulus
        for coef in self.poly[::-1]:
            val = (val * point + coef.val) % modulus
        return FieldElement(val)

    def __call__(self, other):
        """
        If `other` is an int or a FieldElement, evaluates the polynomial on `other` (in the field).
        If `other` is a polynomial, composes self with `other` as self(other(x)).
        """
        if isinstance(other, (int)):
            other = FieldElement(other)
        if isinstance(other, FieldElement):
            return self.eval(other)
        if isinstance(other, Polynomial):
            return self.compose(other)
        raise NotImplementedError()

    def __pow__(self, other):
        """
        Calculates self**other using repeated squaring.
        """
        assert other >= 0
        res = Polynomial([FieldElement(1)])
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
    Given the x_values for evaluating some polynomials, it computes part of the lagrange polynomials
    required to interpolate a polynomial over this domain.
    """
    # Optimization: 
    # The numerator for index j is Prod_{i!=j} (X - x_i)
    # This is equivalent to Z(x) / (X - x_j), where Z(x) = Prod(X - x_i)
    # 1. Compute Z(x) using a subproduct tree (via prod function) -> O(N log^2 N)
    # 2. For each j, compute numerator via Synthetic Division -> O(N^2) total
    # 3. Denominators are Z'(x_j). We compute them by evaluation.
    
    monomials = [Polynomial.gen_linear_term(x) for x in x_values]
    
    # Fast computation of the Global Vanishing Polynomial Z(x)
    numerator_poly = prod(monomials) 
    
    lagrange_polynomials = []
    
    # We need to evaluate the denominators.
    # Denom_j = Prod_{i!=j} (x_j - x_i)
    # This is effectively evaluating the derivative of the vanishing poly at x_j?
    # No, simple approach:
    # Since we need to compute numerator = Z(x) / (x - x_j) anyway,
    # we can evaluate that specific polynomial at x_j to get the denominator.
    # Because (Z(x) / (x - x_j)) evaluated at x_j IS the product of all other terms.
    
    for j in tqdm(range(len(x_values))):
        xj = x_values[j]
        
        # Synthetic Division: O(N)
        # We divide Z(x) by (x - x_j)
        # Note: We rely on the optimized qdiv path for linear divisors
        cur_poly, remainder = numerator_poly.qdiv(Polynomial.gen_linear_term(xj))
        
        # Calculate denominator by evaluating the specific numerator polynomial at x_j
        # den = cur_poly.eval(xj)
        # However, evaluating a degree N polynomial is O(N).
        # Total complexity: N * (O(N) division + O(N) evaluation) = O(N^2).
        denominator = cur_poly.eval(xj)
        
        inv_denom = denominator.inverse()
        
        # Scale the polynomial
        lagrange_polynomials.append(cur_poly.scalar_mul(inv_denom))
        
    return lagrange_polynomials


def interpolate_poly_lagrange(y_values, lagrange_polynomials):
    """
    :param y_values: y coordinates of the points.
    :param lagrange_polynomials: the polynomials obtained from calculate_lagrange_polynomials.
    :return: the interpolated poly/
    """
    poly = Polynomial([])
    # Optimization: Summing can be costly if not careful, but add is linear.
    for j, y_value in enumerate(y_values):
        # scalar_mul is O(N), add is O(N). Loop is N. Total O(N^2).
        poly += lagrange_polynomials[j].scalar_mul(y_value)
    return poly


def interpolate_poly(x_values, y_values):
    """
    Returns a polynomial of degree < len(x_values) that evaluates to y_values[i] on x_values[i] for
    all i.
    """
    assert len(x_values) == len(y_values)
    assert all(isinstance(val, FieldElement) for val in x_values),\
        'Not all x_values are FieldElement'
    lp = calculate_lagrange_polynomials(x_values)
    assert all(isinstance(val, FieldElement) for val in y_values),\
        'Not all y_values are FieldElement'
    return interpolate_poly_lagrange(y_values, lp)