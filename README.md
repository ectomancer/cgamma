# cgamma
m_tgamma ported from C, refactored to complex gamma function with no complex logic and no bounds checking for imaginary part.
 Some error codes not ported.<br>
 assert statements not implemented.<br>
 Some C functions ported: isfinite, sin, cos.<br>
 Some C functions and constants imported from math module: fmod, fabs, copysign, nan, inf.

 Two logic errors corrected in lanczos_sum but gamma function is less accurate.

 Complex domain failing, real domain working.
