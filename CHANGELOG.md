# Changelog

v1.2.0

* Added `Noise2DWithDerivatives`: 2D noise value (bit-identical to
`Noise2D`) plus analytic x/y partial derivatives, continuous across lattice
boundaries
* Added `Curl2D`/`CurlNoise2D`: divergence-free 2D curl (flow) field built
from the noise potential, for particle advection
* Added `Streamline2D`: RK2 streamline integration of the curl field with a
given step size

v1.1.0

## BREAKING CHANGES

* Switched to using int32 internally to keep on par with the original version,
while the interface remains the same the noise values differ from `v1.0.0`

v1.0.0

* Initial version
