# GO Perlin Noise generator [![Go Reference](https://pkg.go.dev/badge/github.com/aquilax/go-perlin.svg)](https://pkg.go.dev/github.com/aquilax/go-perlin)

Adapted for go from [GEGL](http://git.gnome.org/browse/gegl/tree/operations/common/perlin)

Example: `perlin.Noise1D`

    alpha = 2
    beta = 2
    n = 3

![PerlinNoise1D](http://i.imgur.com/Kplg5.png)

Using `perlin.Noise2D` with termbox to generate terrain in terminal:

![PerlinNoise2D](http://i.imgur.com/vPi3n.png)

## Curl (divergence-free) flow field

For particle-flow effects, `Curl2D` builds a divergence-free vector field
from the 2D noise potential `psi`: `v = (d psi/dy, -d psi/dx)`, using
analytic derivatives that are continuous across lattice boundaries, so
streamlines never jump at grid cells. `Noise2DWithDerivatives` exposes the
raw value (bit-identical to `Noise2D`) and partial derivatives,
`CurlNoise2D` adds explicit frequency/octave control, and `Streamline2D`
integrates a polyline along the field with a given step size:

    p := perlin.NewPerlin(2, 2, 3, 100)
    for _, pt := range p.Streamline2D(x0, y0, 0.05, 64) {
        // advect a particle to pt
    }
