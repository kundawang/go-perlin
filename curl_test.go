package perlin

import (
	"math"
	"math/rand"
	"testing"
)

// sameFloat reports whether a and b are equal, treating NaN == NaN as equal.
func sameFloat(a, b float64) bool {
	return a == b || (math.IsNaN(a) && math.IsNaN(b))
}

// The value returned by Noise2DWithDerivatives must be identical to Noise2D.
func Test_Noise2DWithDerivativesMatchesNoise2D(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	r := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		x := r.Float64()*200 - 100
		y := r.Float64()*200 - 100
		value, _, _ := p.Noise2DWithDerivatives(x, y)
		if want := p.Noise2D(x, y); value != want {
			t.Fatalf("value mismatch at (%v, %v): got %v, want %v", x, y, value, want)
		}
	}
}

// Analytic derivatives must match central differences of Noise2D.
func Test_Noise2DDerivativesCentralDifference(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	r := rand.New(rand.NewSource(7))
	const h = 1e-4
	const tol = 1e-3
	for i := 0; i < 500; i++ {
		x := r.Float64()*20 - 10
		y := r.Float64()*20 - 10
		_, ddx, ddy := p.Noise2DWithDerivatives(x, y)
		numDDX := (p.Noise2D(x+h, y) - p.Noise2D(x-h, y)) / (2 * h)
		numDDY := (p.Noise2D(x, y+h) - p.Noise2D(x, y-h)) / (2 * h)
		if math.Abs(ddx-numDDX) > tol {
			t.Errorf("ddx mismatch at (%v, %v): analytic %v, numeric %v", x, y, ddx, numDDX)
		}
		if math.Abs(ddy-numDDY) > tol {
			t.Errorf("ddy mismatch at (%v, %v): analytic %v, numeric %v", x, y, ddy, numDDY)
		}
	}
}

// Value and derivatives must be continuous across lattice (integer)
// boundaries in both axes.
func Test_Noise2DDerivativesLatticeContinuity(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	const eps = 1e-6
	const tol = 1e-4
	r := rand.New(rand.NewSource(11))
	for i := 0; i < 200; i++ {
		k := float64(r.Intn(40) - 20) // integer lattice coordinate
		y := r.Float64()*20 - 10

		vl, dxl, dyl := p.Noise2DWithDerivatives(k-eps, y)
		vr, dxr, dyr := p.Noise2DWithDerivatives(k+eps, y)
		if math.Abs(vl-vr) > tol || math.Abs(dxl-dxr) > tol || math.Abs(dyl-dyr) > tol {
			t.Errorf("discontinuity across x=%v at y=%v: dv=%v ddx=%v ddy=%v",
				k, y, math.Abs(vl-vr), math.Abs(dxl-dxr), math.Abs(dyl-dyr))
		}

		x := r.Float64()*20 - 10
		vb, dxb, dyb := p.Noise2DWithDerivatives(x, k-eps)
		vt, dxt, dyt := p.Noise2DWithDerivatives(x, k+eps)
		if math.Abs(vb-vt) > tol || math.Abs(dxb-dxt) > tol || math.Abs(dyb-dyt) > tol {
			t.Errorf("discontinuity across y=%v at x=%v: dv=%v ddx=%v ddy=%v",
				k, x, math.Abs(vb-vt), math.Abs(dxb-dxt), math.Abs(dyb-dyt))
		}
	}
}

// The curl field must be (numerically) divergence-free.
func Test_Curl2DDivergenceFree(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	r := rand.New(rand.NewSource(13))
	const h = 1e-4
	const tol = 5e-4
	for i := 0; i < 500; i++ {
		// keep clear of lattice boundaries so the central difference of the
		// (only C1) field converges
		x := float64(r.Intn(20)-10) + 0.2 + 0.6*r.Float64()
		y := float64(r.Intn(20)-10) + 0.2 + 0.6*r.Float64()
		vxR, _ := p.Curl2D(x+h, y)
		vxL, _ := p.Curl2D(x-h, y)
		_, vyT := p.Curl2D(x, y+h)
		_, vyB := p.Curl2D(x, y-h)
		div := (vxR-vxL)/(2*h) + (vyT-vyB)/(2*h)
		if math.Abs(div) > tol {
			t.Errorf("non-zero divergence at (%v, %v): %v", x, y, div)
		}
	}
}

// Repeated evaluation at the same point must be deterministic.
func Test_Curl2DDeterministic(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	r := rand.New(rand.NewSource(17))
	for i := 0; i < 100; i++ {
		x := r.Float64()*100 - 50
		y := r.Float64()*100 - 50
		v1, ddx1, ddy1 := p.Noise2DWithDerivatives(x, y)
		v2, ddx2, ddy2 := p.Noise2DWithDerivatives(x, y)
		if v1 != v2 || ddx1 != ddx2 || ddy1 != ddy2 {
			t.Fatalf("non-deterministic derivatives at (%v, %v)", x, y)
		}
		cx1, cy1 := p.Curl2D(x, y)
		cx2, cy2 := p.Curl2D(x, y)
		if cx1 != cx2 || cy1 != cy2 {
			t.Fatalf("non-deterministic curl at (%v, %v)", x, y)
		}
	}
}

// Non-finite coordinates have a defined result: NaN outputs.
func Test_DerivativesNonFiniteCoordinates(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	bad := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, b := range bad {
		for _, pt := range [][2]float64{{b, 0.5}, {0.5, b}, {b, b}} {
			v, ddx, ddy := p.Noise2DWithDerivatives(pt[0], pt[1])
			if !math.IsNaN(v) || !math.IsNaN(ddx) || !math.IsNaN(ddy) {
				t.Errorf("expected NaN outputs at (%v, %v), got %v %v %v", pt[0], pt[1], v, ddx, ddy)
			}
			vx, vy := p.Curl2D(pt[0], pt[1])
			if !math.IsNaN(vx) || !math.IsNaN(vy) {
				t.Errorf("expected NaN curl at (%v, %v), got %v %v", pt[0], pt[1], vx, vy)
			}
			// the simulation-facing API maps invalid input to no flow
			vx, vy = p.CurlNoise2D(pt[0], pt[1], 1, 3)
			if vx != 0 || vy != 0 {
				t.Errorf("expected zero curl at (%v, %v), got %v %v", pt[0], pt[1], vx, vy)
			}
		}
	}
}

// Zero and negative frequency and non-positive octaves yield the zero
// vector; excessive octave counts are clamped to a finite result.
func Test_CurlNoise2DFrequencyAndOctaves(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	for _, f := range []float64{0, -1, -0.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if vx, vy := p.CurlNoise2D(1.3, 2.7, f, 3); vx != 0 || vy != 0 {
			t.Errorf("expected zero vector for frequency %v, got %v %v", f, vx, vy)
		}
	}
	for _, oct := range []int32{0, -1, -100} {
		if vx, vy := p.CurlNoise2D(1.3, 2.7, 1, oct); vx != 0 || vy != 0 {
			t.Errorf("expected zero vector for octaves %v, got %v %v", oct, vx, vy)
		}
	}
	// excessive octaves are clamped, result must stay finite and equal to
	// the MaxOctaves evaluation
	vx, vy := p.CurlNoise2D(1.3, 2.7, 1.5, 1<<20)
	if !isFinite(vx) || !isFinite(vy) {
		t.Fatalf("expected finite curl for huge octave count, got %v %v", vx, vy)
	}
	wx, wy := p.CurlNoise2D(1.3, 2.7, 1.5, MaxOctaves)
	if vx != wx || vy != wy {
		t.Errorf("octave clamping mismatch: %v %v vs %v %v", vx, vy, wx, wy)
	}
	// a generator created with an excessive octave count is clamped too
	big := NewPerlin(2, 2, 1<<20, seed)
	if _, ddx, ddy := big.Noise2DWithDerivatives(1.3, 2.7); !isFinite(ddx) || !isFinite(ddy) {
		t.Errorf("expected finite derivatives for huge generator octave count, got %v %v", ddx, ddy)
	}
	// zero/negative beta (the generator's frequency scaling) stays finite
	for _, b := range []float64{0, -2} {
		bp := NewPerlin(2, b, 3, seed)
		if vx, vy := bp.Curl2D(1.3, 2.7); !isFinite(vx) || !isFinite(vy) {
			t.Errorf("expected finite curl for beta=%v, got %v %v", b, vx, vy)
		}
	}
}

// Extremely large coordinates: defined, deterministic, and the value still
// matches Noise2D exactly.
func Test_DerivativesHugeCoordinates(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	for _, c := range []float64{1e10, 1e18, 1e300, -1e300} {
		v1, _, _ := p.Noise2DWithDerivatives(c, c)
		v2, _, _ := p.Noise2DWithDerivatives(c, c)
		if !sameFloat(v1, v2) {
			t.Errorf("non-deterministic value at huge coordinate %v: %v vs %v", c, v1, v2)
		}
		if want := p.Noise2D(c, c); !sameFloat(v1, want) {
			t.Errorf("value mismatch vs Noise2D at %v: got %v, want %v", c, v1, want)
		}
	}
	// coordinates that overflow float64 when scaled by the frequency yield
	// the zero vector from the simulation-facing API
	if vx, vy := p.CurlNoise2D(1e300, 1e300, 1e300, 3); vx != 0 || vy != 0 {
		t.Errorf("expected zero vector for overflowing coordinates, got %v %v", vx, vy)
	}
}

// A zero field vector (here: zero step / zero octaves) leaves the particle
// at rest, and invalid arguments return nil.
func Test_Streamline2DEdgeCases(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)

	// zero step: the particle never moves
	pts := p.Streamline2D(1.3, 2.7, 0, 8)
	if len(pts) != 9 {
		t.Fatalf("expected 9 points, got %d", len(pts))
	}
	for i, pt := range pts {
		if pt[0] != 1.3 || pt[1] != 2.7 {
			t.Errorf("point %d moved under zero step: %v", i, pt)
		}
	}

	// invalid arguments
	if got := p.Streamline2D(1, 1, 0.1, 0); got != nil {
		t.Errorf("expected nil for steps=0, got %v", got)
	}
	if got := p.Streamline2D(1, 1, 0.1, -5); got != nil {
		t.Errorf("expected nil for negative steps, got %v", got)
	}
	if got := p.Streamline2D(math.NaN(), 1, 0.1, 4); got != nil {
		t.Errorf("expected nil for NaN start, got %v", got)
	}
	if got := p.Streamline2D(1, math.Inf(1), 0.1, 4); got != nil {
		t.Errorf("expected nil for Inf start, got %v", got)
	}
	if got := p.Streamline2D(1, 1, math.NaN(), 4); got != nil {
		t.Errorf("expected nil for NaN step, got %v", got)
	}
}

// A normal streamline: correct length, finite points, bounded step length,
// deterministic.
func Test_Streamline2DIntegration(t *testing.T) {
	p := NewPerlin(2, 2, 3, seed)
	const steps = 64
	const step = 0.05
	pts := p.Streamline2D(3.7, -1.2, step, steps)
	if len(pts) != steps+1 {
		t.Fatalf("expected %d points, got %d", steps+1, len(pts))
	}
	if pts[0] != [2]float64{3.7, -1.2} {
		t.Errorf("first point is not the start: %v", pts[0])
	}
	maxV := 0.
	for i := 0; i < 100; i++ { // rough upper bound of the field magnitude
		vx, vy := p.Curl2D(float64(i)/10, float64(i)/7)
		if m := math.Hypot(vx, vy); m > maxV {
			maxV = m
		}
	}
	for i, pt := range pts {
		if !isFinite(pt[0]) || !isFinite(pt[1]) {
			t.Fatalf("non-finite point %d: %v", i, pt)
		}
		if i > 0 {
			d := math.Hypot(pt[0]-pts[i-1][0], pt[1]-pts[i-1][1])
			if d > step*maxV*2 {
				t.Errorf("step %d too long: %v", i, d)
			}
		}
	}
	again := p.Streamline2D(3.7, -1.2, step, steps)
	for i := range pts {
		if pts[i] != again[i] {
			t.Fatalf("non-deterministic streamline at point %d: %v vs %v", i, pts[i], again[i])
		}
	}
}

func Benchmark_Curl2D(b *testing.B) {
	p := NewPerlin(2, 2, 3, seed)
	for n := 0; n < b.N; n++ {
		p.Curl2D(10, 10)
	}
}
