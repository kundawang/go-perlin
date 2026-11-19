// This file adds analytic derivatives of the 2D noise and a divergence-free
// curl (flow) field built on top of them, suitable e.g. for advecting
// particles along smooth streamlines. It does not change Noise2D in any way:
// the value returned by Noise2DWithDerivatives is computed with exactly the
// same operations, in the same order, as Noise2D.
package perlin

import "math"

// MaxOctaves is the maximum number of octaves evaluated by the
// derivative-based APIs (Noise2DWithDerivatives, Curl2D, CurlNoise2D and
// Streamline2D). Beyond 64 octaves the per-octave amplitude and frequency
// factors are no longer representable in float64, so a generator created
// with a larger octave count is clamped to MaxOctaves instead of producing
// Inf/NaN. As long as the generator's n <= MaxOctaves, the value returned by
// Noise2DWithDerivatives is bit-identical to Noise2D.
const MaxOctaves = 64

// dsCurve is the first derivative of the smoothing curve sCurve,
// s'(t) = 6t(1-t). It is zero at t=0 and t=1, which makes the noise (and
// therefore its analytic gradient) continuous across lattice boundaries.
func dsCurve(t float64) float64 {
	return 6. * t * (1. - t)
}

// noise2WithGradient evaluates noise2 together with its analytic partial
// derivatives. The value is computed with the same operations as noise2, so
// it is bit-identical to it.
func (p *Perlin) noise2WithGradient(vec [2]float64) (val, gx, gy float64) {
	t := vec[0] + N
	bx0 := int32(t) & BM
	bx1 := (bx0 + 1) & BM
	rx0 := t - float64(int32(t))
	rx1 := rx0 - 1.

	t = vec[1] + N
	by0 := int32(t) & BM
	by1 := (by0 + 1) & BM
	ry0 := t - float64(int32(t))
	ry1 := ry0 - 1.

	i := p.p[bx0]
	j := p.p[bx1]

	b00 := p.p[i+by0]
	b10 := p.p[j+by0]
	b01 := p.p[i+by1]
	b11 := p.p[j+by1]

	sx := sCurve(rx0)
	sy := sCurve(ry0)
	dsx := dsCurve(rx0)
	dsy := dsCurve(ry0)

	g00 := p.g2[b00]
	g10 := p.g2[b10]
	g01 := p.g2[b01]
	g11 := p.g2[b11]

	u := at2(rx0, ry0, g00)
	v := at2(rx1, ry0, g10)
	a := lerp(sx, u, v)
	dadx := dsx*(v-u) + lerp(sx, g00[0], g10[0])
	dady := lerp(sx, g00[1], g10[1])

	u = at2(rx0, ry1, g01)
	v = at2(rx1, ry1, g11)
	b := lerp(sx, u, v)
	dbdx := dsx*(v-u) + lerp(sx, g01[0], g11[0])
	dbdy := lerp(sx, g01[1], g11[1])

	val = lerp(sy, a, b)
	gx = lerp(sy, dadx, dbdx)
	gy = dsy*(b-a) + lerp(sy, dady, dbdy)
	return val, gx, gy
}

// noise2DWithDerivatives accumulates the value and gradient over the given
// number of octaves using the generator's alpha/beta rules. octaves is
// clamped to MaxOctaves; octaves <= 0 yields zero value and zero gradient.
func (p *Perlin) noise2DWithDerivatives(x, y float64, octaves int32) (value, ddx, ddy float64) {
	if octaves > MaxOctaves {
		octaves = MaxOctaves
	}
	var scale float64 = 1
	var freq float64 = 1
	var val, gx, gy float64
	var i int32
	px := [2]float64{x, y}

	for i = 0; i < octaves; i++ {
		val, gx, gy = p.noise2WithGradient(px)
		value += val / scale
		ddx += freq * gx / scale
		ddy += freq * gy / scale
		scale *= p.alpha
		freq *= p.beta
		px[0] *= p.beta
		px[1] *= p.beta
	}
	return value, ddx, ddy
}

// Noise2DWithDerivatives generates the 2-dimensional Perlin noise value at
// (x, y) together with its analytic partial derivatives d/dx and d/dy.
//
// The value is bit-identical to Noise2D(x, y) as long as the generator's
// octave count does not exceed MaxOctaves; Noise2D itself is untouched.
// The derivatives are analytic (not finite differences) and continuous
// across lattice boundaries, and repeated calls at the same point return
// identical results.
//
// If x or y is NaN or infinite, all three return values are NaN. Extremely
// large finite coordinates behave exactly like Noise2D (the same lattice
// arithmetic is used), which may yield non-finite results once intermediate
// values overflow float64.
func (p *Perlin) Noise2DWithDerivatives(x, y float64) (value, ddx, ddy float64) {
	if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
		return math.NaN(), math.NaN(), math.NaN()
	}
	return p.noise2DWithDerivatives(x, y, p.n)
}

// Curl2D returns a divergence-free 2D vector field derived from the noise
// potential psi = Noise2D: v = (d psi/dy, -d psi/dx). The divergence
// d vx/dx + d vy/dy is analytically zero, so streamlines of this field are
// smooth, non-intersecting and free of sinks and sources — suitable for
// flowing particle effects. Seed, alpha, beta and the octave count follow
// the existing generator rules. Non-finite coordinates yield (NaN, NaN);
// a point where the noise gradient vanishes yields the zero vector (0, 0).
func (p *Perlin) Curl2D(x, y float64) (vx, vy float64) {
	_, ddx, ddy := p.Noise2DWithDerivatives(x, y)
	return ddy, -ddx
}

// CurlNoise2D samples the divergence-free curl field at (x, y) with an
// explicit sampling frequency and octave count, using the generator's seed,
// alpha and beta. It is the curl of the potential psi(frequency*x,
// frequency*y) summed over the given octaves, i.e. Curl2D scaled by
// frequency.
//
// Defined edge-case results: frequency <= 0 or non-finite, non-finite
// coordinates, octaves <= 0, or a sampling coordinate overflowing float64
// all return the zero vector (0, 0). octaves is clamped to MaxOctaves.
func (p *Perlin) CurlNoise2D(x, y, frequency float64, octaves int32) (vx, vy float64) {
	if octaves <= 0 || frequency <= 0 || math.IsNaN(frequency) || math.IsInf(frequency, 0) ||
		math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
		return 0, 0
	}
	sx := x * frequency
	sy := y * frequency
	if math.IsNaN(sx) || math.IsNaN(sy) || math.IsInf(sx, 0) || math.IsInf(sy, 0) {
		return 0, 0
	}
	_, ddx, ddy := p.noise2DWithDerivatives(sx, sy, octaves)
	return frequency * ddy, -frequency * ddx
}

// Streamline2D integrates the Curl2D flow field starting at (x, y), taking
// the given number of RK2 (midpoint) steps of the given step size, and
// returns the resulting polyline of steps+1 points including the start
// point. A negative step size integrates backwards along the field.
//
// Defined edge-case results: steps <= 0 or a non-finite start point or step
// size return nil. A zero field vector leaves the particle at rest (the
// point is repeated). If the field ever becomes non-finite mid-integration
// (e.g. the trace runs into overflowing coordinates), the polyline is
// truncated at the last finite point.
func (p *Perlin) Streamline2D(x, y, step float64, steps int) [][2]float64 {
	if steps <= 0 || math.IsNaN(x) || math.IsNaN(y) || math.IsNaN(step) ||
		math.IsInf(x, 0) || math.IsInf(y, 0) || math.IsInf(step, 0) {
		return nil
	}
	pts := make([][2]float64, 1, steps+1)
	pts[0] = [2]float64{x, y}
	cur := pts[0]
	for i := 0; i < steps; i++ {
		vx, vy := p.Curl2D(cur[0], cur[1])
		if !isFinite(vx) || !isFinite(vy) {
			return pts
		}
		mx := cur[0] + 0.5*step*vx
		my := cur[1] + 0.5*step*vy
		vx, vy = p.Curl2D(mx, my)
		if !isFinite(vx) || !isFinite(vy) {
			return pts
		}
		cur[0] += step * vx
		cur[1] += step * vy
		pts = append(pts, cur)
	}
	return pts
}

func isFinite(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}
