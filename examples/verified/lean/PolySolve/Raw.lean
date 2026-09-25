import PolySolve.Poly

/-!
# Polynomials at the Python boundary, and helpers for building them

A polynomial crosses to Python and back as a list of `(e1, e2, e3, e4, e5, c)`:
the exponents of `x1` … `x5` and the coefficient, in any order. `polyOfRaw`
sorts and merges on the way in, so every check sees one canonical form.

The helpers below build polynomials and are not proved correct: whatever they
produce is only data, and the checks in `Elimination.lean` decide whether it is
what it is claimed to be.
-/

namespace PolySolve

/-- A term as it crosses to Python: the exponents of `x1` … `x5`, then the coefficient. -/
abbrev RawTerm : Type := Nat × Nat × Nat × Nat × Nat × Int

abbrev RawPoly : Type := List RawTerm

def termOfRaw (t : RawTerm) : Term :=
  let m : Mono := ⟨t.1, t.2.1, t.2.2.1, t.2.2.2.1, t.2.2.2.2.1⟩
  ⟨m.key, m, t.2.2.2.2.2⟩

/-- Sort and merge terms, dropping zero coefficients. -/
def normalize (p : Poly) : Poly :=
  p.foldr (fun t acc => if t.coeff = 0 then acc else addPoly [t] acc) []

def polyOfRaw (p : RawPoly) : Poly := normalize (p.map termOfRaw)

def polysOfRaw (ps : List RawPoly) : List Poly := ps.map polyOfRaw

def rawOfPoly (p : Poly) : RawPoly :=
  p.map fun t => (t.mono.e1, t.mono.e2, t.mono.e3, t.mono.e4, t.mono.e5, t.coeff)

section Eval

variable {R : Type} [Lean.Grind.CommRing R] (x1 x2 x3 x4 x5 : R)

/-- Normalizing keeps a polynomial's value: a raw polynomial means the sum of its terms. -/
theorem eval_normalize (p : Poly) : evalPoly x1 x2 x3 x4 x5 (normalize p) = evalPoly x1 x2 x3 x4 x5 p := by
  induction p with
  | nil => rfl
  | cons t p ih =>
    simp only [normalize, List.foldr_cons] at ih ⊢
    split
    · rename_i h; simp only [evalPoly_cons, h, ih, Lean.Grind.Ring.intCast_zero]; grind
    · rw [eval_addPoly, ih]; simp only [evalPoly_cons, evalPoly_nil]; grind

end Eval

/-! ### Building polynomials -/

/-- The variable `x_i`, for `i` from 1 to 5. -/
def var (i : Nat) : Poly :=
  let m : Mono := ⟨if i = 1 then 1 else 0, if i = 2 then 1 else 0, if i = 3 then 1 else 0,
    if i = 4 then 1 else 0, if i = 5 then 1 else 0⟩
  [⟨m.key, m, 1⟩]

def constPoly (c : Int) : Poly := normalize [⟨0, Mono.one, c⟩]

def Mono.exp (m : Mono) (i : Nat) : Nat :=
  if i = 1 then m.e1 else if i = 2 then m.e2 else if i = 3 then m.e3 else if i = 4 then m.e4 else m.e5

def Mono.dropExp (m : Mono) (i : Nat) : Mono :=
  ⟨if i = 1 then 0 else m.e1, if i = 2 then 0 else m.e2, if i = 3 then 0 else m.e3,
    if i = 4 then 0 else m.e4, if i = 5 then 0 else m.e5⟩

/-- The coefficient of `x_i ^ d` in `p`, as a polynomial in the other variables. -/
def coefficient (p : Poly) (i d : Nat) : Poly :=
  normalize (p.filterMap fun t =>
    if t.mono.exp i = d then let m := t.mono.dropExp i; some ⟨m.key, m, t.coeff⟩ else none)

/-- The highest power of `x_i` in `p`. -/
def degreeIn (p : Poly) (i : Nat) : Nat := p.foldr (fun t acc => max (t.mono.exp i) acc) 0

/-- `p` with `q` substituted for `x_i`. -/
def substVar (p : Poly) (i : Nat) (q : Poly) : Poly :=
  compose p (if i = 1 then q else var 1) (if i = 2 then q else var 2) (if i = 3 then q else var 3)
    (if i = 4 then q else var 4) (if i = 5 then q else var 5)

end PolySolve
