import math

# e = math.exp(1)
e = 2.718281828459045
# π = math.pi
π = 3.141592653589793
EMPTY_SUM = 0
NGAMMA_INTEGRAL = 23
LANCZOS_N = 13
NGAMMA_INTEGRAL = 23
lanczos_g = 6.024680040776729583740234375
lanczos_g_minus_half = 5.524680040776729583740234375
lanczos_num_coeffs = [
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408
    ]

# Denominator is x*(x+1)*...*(x+LANCZOS_N-2).
lanczos_den_coeffs = [
    0, 39916800, 120543840, 150917976, 105258076, 45995730,
    13339535, 2637558, 357423, 32670, 1925, 66, 1]

# Values for gamma for small positive integers, 1 though NGAMMA_INTEGRAL.
gamma_integral = [
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
    3628800, 39916800, 479001600, 6227020800, 87178291200,
    1307674368000, 20922789888000, 355687428096000,
    6402373705728000, 121645100408832000, 2432902008176640000,
    51090942171709440000, 1124000727777607680000,
]

class MathDomainError(Exception):
    pass

class MathRangeError(Exception):
    """Math result not representable."""
    pass

def isfinite(x: float) -> bool:
    """Test if number is finite."""
    if x == float('inf'):
        return False
    if x == -float('inf'):
        return False
    if x == float('NaN'):
        return False
    if x != x:  # Identity by C. K. Young@stackoverflow.com
        return False
    return True

def sin(x: float) -> float:
    """Pure Python sine function of a real angle in radians."""
    return ((e**(1j*x) - e**(-1j*x))/2j).real

def cos(x: float) -> float:
    """Pure Python cosine function of a real angle in radians."""
    return ((e**(1j*x) + e**(-1j*x))/2).real

def sinpi(x: float) -> float:
    y = math.fmod(math.fabs(x), 2)
    n = round(2*y)
    if n == 0:
        r = sin(π*y)
    elif n == 1:
        r = cos(π*(y - 0.5))
    # N.B. -sin(π*(y-1.0)) is *not* equivalent: it would give
    # -0 instead of 0 when y==1.
    elif n == 2:
        r = sin(π*(1 - y))
    elif n == 3:
        r = -cos(π*(y - 1.5))
    elif n == 4:
        r = sin(π*(y - 2))
    else:
        raise MathDomainError
    return math.copysign(1, x)*r

def lanczos_sum(z: complex) -> complex:
    """Lanczos's sum, L_g(x), for positive."""
    num, den = EMPTY_SUM, EMPTY_SUM
    """Evaluate the rational function lanczos_sum(x).  For large
       x, the obvious algorithm risks overflow, so we instead
       rescale the denominator and numerator of the rational
       function by x**(1-LANCZOS_N) and treat this as a
       rational function in 1/x.  This also reduces the error for
       larger x values.  The choice of cutoff point (5.0 below) is
       somewhat arbitrary; in tests, smaller cutoff values than
       this resulted in lower accuracy.
    """
    if z.real < 5:
        for i in range(LANCZOS_N-1, -1, -1):
            num = num*z + lanczos_num_coeffs[i]
            den = den*z + lanczos_den_coeffs[i]
    else:
        for i in range(LANCZOS_N):
            num = num/z + lanczos_num_coeffs[i]
            den = den/z + lanczos_den_coeffs[i]
    return num/den

def gamma(z: complex) -> complex:
    """Complex gamma function.
    m_tgamma ported from C.
    Implementation of the real gamma function.  Kept here to work around
    issues (see e.g. gh-70309) with quality of libm's tgamma/lgamma implementations
    on various platforms (Windows, MacOS).  In extensive but non-exhaustive
    random tests, this function proved accurate to within <= 10 ulps across the
    entire float domain.  Note that accuracy may depend on the quality of the
    system math functions, the pow function in particular.  Special cases
    follow C99 annex F.  The parameters and method are tailored to platforms
    whose double format is the IEEE 754 binary64 format.

    Method: for x > 0.0 we use the Lanczos approximation with parameters N=13
    and g=6.024680040776729583740234375; these parameters are amongst those
    used by the Boost library.  Following Boost (again), we re-express the
    Lanczos sum as a rational function, and compute it that way.  The
    coefficients below were computed independently using MPFR, and have been
    double-checked against the coefficients in the Boost source code.

    For x < 0.0 we use the reflection formula.

    There's one minor tweak that deserves explanation: Lanczos' formula for
    gamma(x) involves computing pow(x+g-0.5, x-0.5) / exp(x+g-0.5).  For many x
    values, x+g-0.5 can be represented exactly.  However, in cases where it
    can't be represented exactly the small error in x+g-0.5 can be magnified
    significantly by the pow and exp calls, especially for large x.  A cheap
    correction is to multiply by (1 + e*g/(x+g-0.5)), where e is the error
    involved in the computation of x+g-0.5 (that is, e = computed value of
    x+g-0.5 - exact value of x+g-0.5).  Here's the proof:

    Correction factor
    -----------------
    Write x+g-0.5 = y-e, where y is exactly representable as an IEEE 754
    double, and e is tiny.  Then:

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) = pow(y-e, x-0.5)/exp(y-e)
     = pow(y, x-0.5)/exp(y) * C,

    where the correction_factor C is given by

     C = pow(1-e/y, x-0.5) * exp(e)

    Since e is tiny, pow(1-e/y, x-0.5) ~ 1-(x-0.5)*e/y, and exp(x) ~ 1+e, so:

     C ~ (1-(x-0.5)*e/y) * (1+e) ~ 1 + e*(y-(x-0.5))/y

    But y-(x-0.5) = g+e, and g+e ~ g.  So we get C ~ 1 + e*g/y, and

     pow(x+g-0.5,x-0.5)/exp(x+g-0.5) ~ pow(y, x-0.5)/exp(y) * (1 + e*g/y),

    Note that for accuracy, when computing r*C it's better to do

     r + e*g/y*r;

    than

     r * (1 + e*g/y);

    since the addition in the latter throws away most of the bits of
    information in e*g/y.
    """
    # Special cases.
    if not isfinite(z.real) and not isfinite(z.imag):
        if z.real != z.real or z.real > 0:
            return z  # gamma(nan) = nan, gamma(inf) = inf
        else:
            return math.nan  # gamma(-inf) = nan, invalid.
    if z == 0:  # gamma(+-0.0) = +-inf, divide-by-zero.
        return math.copysign(math.inf, z.real)

    # Integer arguments.
    if z.real == z.real.__floor__():
        return math.nan  # gamma(n) = nan, invalid for negative integers n.
        if z.real < 0:
            if z.real <= NGAMMA_INTEGRAL:
                return gamma_integral[z.real-1]
    absx = math.fabs(z.real)

    # Tiny arguments:  gamma(x) ~ 1/x for x near 0.
    if absx < 1e-20:
        r = 1/x
        if r == float('inf'):
            return r

    # Large arguments: assuming IEEE 754 doubles, gamma(x) overflows for
    # x>200 and underflows to +-0.0 for x<-200, not a negative integer.
    if absx > 200:
        if z.real < 0:
            return 0/sinpi(z.real)

    y = absx + lanczos_g_minus_half
    # Compute error in sum.
    if absx > lanczos_g_minus_half:
    # The correction can be foiled by an optimizing
    # compiler that (incorrectly) thinks that an expression like
    # a + b - a - b can be optimized to 0.0.
        q = y - absx
        zz = q - lanczos_g_minus_half
    else:
        q = y - lanczos_g_minus_half
        zz = q - absx
    zz *= lanczos_g/y
    if z.real < 0:
        r = -π/sinpi(absx)/absx*e**y/lanczos_sum(absx)
        r -= zz*r
        if absx < 140:
            r /= y**(absx - 0.5)
        else:
            sqrtpow = y**(absx/2 - 0.25)
            r /= sqrtpow
            r /= sqrtpow
    else:
        r = lanczos_sum(absx)/e**y
        r += zz*r
        if absx < 140:
            r *= y**(absx - 0.5)
        else:
            sqrtpow = y**(absx/2 - 0.25)
            r *= sqrtpow
            r *= sqrtpow
    if abs(r) == float('inf'):
        raise MathRangeError
    return r
