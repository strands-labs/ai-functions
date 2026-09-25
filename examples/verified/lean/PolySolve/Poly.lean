/-!
# Integer polynomials in five variables

A polynomial is a list of terms, read as their sum: `evalPoly` fixes what a list
means, in any commutative ring, and every operation below comes with a theorem
that it computes what it should under that reading. The operations are
structurally recursive, so the kernel can run them: a claimed polynomial identity
is checked by `rfl`, and the theorems turn the check into a fact about values.

Terms are kept sorted by `Mono.key` so additions merge in one pass. Sortedness is
for speed only; no theorem assumes it.
-/

namespace PolySolve

open Lean.Grind

-- Core's `CommRing` carries an `Int` cast; make it the coercion `(c : R)`.
attribute [instance] Ring.intCast

/-- An exponent vector in the variables `x1` … `x5`. -/
structure Mono where
  e1 : Nat
  e2 : Nat
  e3 : Nat
  e4 : Nat
  e5 : Nat
deriving DecidableEq, Repr

def Mono.one : Mono := ⟨0, 0, 0, 0, 0⟩

def Mono.mul (m m' : Mono) : Mono :=
  ⟨m.e1 + m'.e1, m.e2 + m'.e2, m.e3 + m'.e3, m.e4 + m'.e4, m.e5 + m'.e5⟩

/-- The sort order of monomials; exponents below 64 give distinct keys. -/
def Mono.key (m : Mono) : Nat :=
  m.e1 + 64 * (m.e2 + 64 * (m.e3 + 64 * (m.e4 + 64 * m.e5)))

/-- A coefficient times a monomial. `key` caches `mono.key`: `evalPoly` never reads it. -/
structure Term where
  key : Nat
  mono : Mono
  coeff : Int
deriving DecidableEq, Repr

def Term.mul (t t' : Term) : Term :=
  ⟨t.key + t'.key, t.mono.mul t'.mono, t.coeff * t'.coeff⟩

/-- A polynomial: the sum of its terms. -/
abbrev Poly := List Term

variable {R : Type} [CommRing R]

def evalMono (x1 x2 x3 x4 x5 : R) (m : Mono) : R :=
  x1 ^ m.e1 * x2 ^ m.e2 * x3 ^ m.e3 * x4 ^ m.e4 * x5 ^ m.e5

def evalPoly (x1 x2 x3 x4 x5 : R) : Poly → R
  | [] => 0
  | t :: p => (t.coeff : R) * evalMono x1 x2 x3 x4 x5 t.mono + evalPoly x1 x2 x3 x4 x5 p

section Eval

variable (x1 x2 x3 x4 x5 : R)

@[simp] theorem evalPoly_nil : evalPoly x1 x2 x3 x4 x5 [] = 0 := rfl

@[simp] theorem evalPoly_cons (t : Term) (p : Poly) :
    evalPoly x1 x2 x3 x4 x5 (t :: p)
      = (t.coeff : R) * evalMono x1 x2 x3 x4 x5 t.mono + evalPoly x1 x2 x3 x4 x5 p := rfl

theorem evalMono_one : evalMono x1 x2 x3 x4 x5 Mono.one = 1 := by
  simp only [evalMono, Mono.one, Semiring.pow_zero, Semiring.mul_one]

theorem evalMono_mul (m m' : Mono) :
    evalMono x1 x2 x3 x4 x5 (m.mul m')
      = evalMono x1 x2 x3 x4 x5 m * evalMono x1 x2 x3 x4 x5 m' := by
  simp only [evalMono, Mono.mul, Semiring.pow_add]
  grind

theorem evalPoly_append (p q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (p ++ q) = evalPoly x1 x2 x3 x4 x5 p + evalPoly x1 x2 x3 x4 x5 q := by
  induction p with
  | nil => simp only [List.nil_append, evalPoly_nil]; grind
  | cons t p ih => simp only [List.cons_append, evalPoly_cons, ih]; grind

/-! ### Addition -/

/-- Split off the terms whose key sorts strictly before `k`. -/
def takeLess (k : Nat) : Poly → Poly × Poly
  | [] => ([], [])
  | t :: qs => if t.key < k then let r := takeLess k qs; (t :: r.1, r.2) else ([], t :: qs)

theorem eval_takeLess (k : Nat) (q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (takeLess k q).1 + evalPoly x1 x2 x3 x4 x5 (takeLess k q).2
      = evalPoly x1 x2 x3 x4 x5 q := by
  induction q with
  | nil => simp only [takeLess, evalPoly_nil]; grind
  | cons t qs ih =>
    by_cases h : t.key < k
    · simp only [takeLess, if_pos h, evalPoly_cons]; grind
    · simp only [takeLess, if_neg h, evalPoly_nil]; grind

/-- The sum of two polynomials, merged in key order. -/
def addPoly : Poly → Poly → Poly
  | [], q => q
  | t :: ps, q =>
      let s := takeLess t.key q
      s.1 ++ (match s.2 with
              | [] => t :: ps
              | t' :: qs =>
                  if t.key == t'.key && t.mono == t'.mono then
                    (if t.coeff + t'.coeff = 0 then addPoly ps qs
                     else ⟨t.key, t.mono, t.coeff + t'.coeff⟩ :: addPoly ps qs)
                  else t :: addPoly ps (t' :: qs))

theorem eval_addPoly : ∀ (p q : Poly),
    evalPoly x1 x2 x3 x4 x5 (addPoly p q) = evalPoly x1 x2 x3 x4 x5 p + evalPoly x1 x2 x3 x4 x5 q := by
  intro p
  induction p with
  | nil => intro q; simp only [addPoly, evalPoly_nil]; grind
  | cons t ps ih =>
    intro q
    rw [← eval_takeLess x1 x2 x3 x4 x5 t.key q]
    rcases hrest : (takeLess t.key q).2 with _ | ⟨t', qs⟩
    · simp only [addPoly, hrest, evalPoly_append, evalPoly_nil, evalPoly_cons]; grind
    · simp only [addPoly, hrest]
      by_cases hm : (t.key == t'.key && t.mono == t'.mono) = true
      · rw [if_pos hm]
        have hmono : t.mono = t'.mono := by simp_all
        by_cases hc : t.coeff + t'.coeff = 0
        · have hz : (t.coeff : R) + (t'.coeff : R) = 0 := by
            rw [← Ring.intCast_add, hc, Ring.intCast_zero]
          rw [if_pos hc, evalPoly_append, ih qs]
          simp only [evalPoly_cons, ← hmono]
          grind
        · rw [if_neg hc, evalPoly_append]
          simp only [evalPoly_cons, ih qs, Ring.intCast_add, ← hmono]
          grind
      · rw [if_neg hm, evalPoly_append]
        simp only [evalPoly_cons, ih (t' :: qs)]
        grind

/-! ### Scaling, multiplication, negation -/

def scaleTerm (t : Term) : Poly → Poly
  | [] => []
  | t' :: q => t.mul t' :: scaleTerm t q

theorem eval_scaleTerm (t : Term) (q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (scaleTerm t q)
      = (t.coeff : R) * evalMono x1 x2 x3 x4 x5 t.mono * evalPoly x1 x2 x3 x4 x5 q := by
  induction q with
  | nil => simp only [scaleTerm, evalPoly_nil]; grind
  | cons t' q ih =>
    simp only [scaleTerm, evalPoly_cons, ih, Term.mul, evalMono_mul, Ring.intCast_mul]; grind

def scaleInt (c : Int) : Poly → Poly
  | [] => []
  | t :: q => ⟨t.key, t.mono, c * t.coeff⟩ :: scaleInt c q

theorem eval_scaleInt (c : Int) (q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (scaleInt c q) = (c : R) * evalPoly x1 x2 x3 x4 x5 q := by
  induction q with
  | nil => simp only [scaleInt, evalPoly_nil]; grind
  | cons t q ih => simp only [scaleInt, evalPoly_cons, ih, Ring.intCast_mul]; grind

def mulPoly : Poly → Poly → Poly
  | [], _ => []
  | t :: ps, q => addPoly (scaleTerm t q) (mulPoly ps q)

theorem eval_mulPoly (p q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (mulPoly p q) = evalPoly x1 x2 x3 x4 x5 p * evalPoly x1 x2 x3 x4 x5 q := by
  induction p with
  | nil => simp only [mulPoly, evalPoly_nil]; grind
  | cons t ps ih => simp only [mulPoly, eval_addPoly, eval_scaleTerm, ih, evalPoly_cons]; grind

def negPoly : Poly → Poly
  | [] => []
  | t :: p => ⟨t.key, t.mono, -t.coeff⟩ :: negPoly p

theorem eval_negPoly (p : Poly) : evalPoly x1 x2 x3 x4 x5 (negPoly p) = -evalPoly x1 x2 x3 x4 x5 p := by
  induction p with
  | nil => simp only [negPoly, evalPoly_nil]; grind
  | cons t p ih => simp only [negPoly, evalPoly_cons, ih, Ring.intCast_neg]; grind

def subPoly (p q : Poly) : Poly := addPoly p (negPoly q)

theorem eval_subPoly (p q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (subPoly p q) = evalPoly x1 x2 x3 x4 x5 p - evalPoly x1 x2 x3 x4 x5 q := by
  simp only [subPoly, eval_addPoly, eval_negPoly]; grind

/-- Two polynomials whose difference normalises to `[]` agree everywhere: the step
from a `rfl` on term lists to a fact about values. -/
theorem eval_eq_of_subPoly_nil {p q : Poly} (h : subPoly p q = []) :
    evalPoly x1 x2 x3 x4 x5 p = evalPoly x1 x2 x3 x4 x5 q := by
  have hs := eval_subPoly x1 x2 x3 x4 x5 p q
  rw [h, evalPoly_nil] at hs
  grind

/-! ### Powers and substitution -/

def onePoly : Poly := [⟨0, Mono.one, 1⟩]

theorem eval_onePoly : evalPoly x1 x2 x3 x4 x5 onePoly = 1 := by
  simp only [onePoly, evalPoly_cons, evalPoly_nil, evalMono_one, Ring.intCast_one]; grind

def powPoly (p : Poly) : Nat → Poly
  | 0 => onePoly
  | 1 => p
  | n + 2 => mulPoly p (powPoly p (n + 1))

theorem eval_powPoly (p : Poly) (n : Nat) :
    evalPoly x1 x2 x3 x4 x5 (powPoly p n) = evalPoly x1 x2 x3 x4 x5 p ^ n := by
  induction n with
  | zero => simp only [powPoly, eval_onePoly, Semiring.pow_zero]
  | succ n ih =>
    match n with
    | 0 => simp only [powPoly]; grind
    | k + 1 => simp only [powPoly] at ih ⊢; rw [eval_mulPoly, ih, Semiring.pow_succ]; grind

/-- Substitute `q1` … `q5` for the variables of one monomial. -/
def substMono (m : Mono) (q1 q2 q3 q4 q5 : Poly) : Poly :=
  mulPoly (powPoly q1 m.e1) (mulPoly (powPoly q2 m.e2)
    (mulPoly (powPoly q3 m.e3) (mulPoly (powPoly q4 m.e4) (powPoly q5 m.e5))))

theorem eval_substMono (m : Mono) (q1 q2 q3 q4 q5 : Poly) :
    evalPoly x1 x2 x3 x4 x5 (substMono m q1 q2 q3 q4 q5)
      = evalMono (evalPoly x1 x2 x3 x4 x5 q1) (evalPoly x1 x2 x3 x4 x5 q2)
          (evalPoly x1 x2 x3 x4 x5 q3) (evalPoly x1 x2 x3 x4 x5 q4)
          (evalPoly x1 x2 x3 x4 x5 q5) m := by
  simp only [substMono, eval_mulPoly, eval_powPoly, evalMono]; grind

/-- Substitute `q1` … `q5` for the variables of a polynomial. -/
def compose : Poly → Poly → Poly → Poly → Poly → Poly → Poly
  | [], _, _, _, _, _ => []
  | t :: p, q1, q2, q3, q4, q5 =>
      addPoly (scaleInt t.coeff (substMono t.mono q1 q2 q3 q4 q5)) (compose p q1 q2 q3 q4 q5)

theorem eval_compose (p q1 q2 q3 q4 q5 : Poly) :
    evalPoly x1 x2 x3 x4 x5 (compose p q1 q2 q3 q4 q5)
      = evalPoly (evalPoly x1 x2 x3 x4 x5 q1) (evalPoly x1 x2 x3 x4 x5 q2)
          (evalPoly x1 x2 x3 x4 x5 q3) (evalPoly x1 x2 x3 x4 x5 q4)
          (evalPoly x1 x2 x3 x4 x5 q5) p := by
  induction p with
  | nil => simp only [compose, evalPoly_nil]
  | cons t p ih => simp only [compose, eval_addPoly, eval_scaleInt, eval_substMono, ih, evalPoly_cons]

/-! ### Systems of equations -/

/-- Every polynomial in the list vanishes at the point. -/
def AllZero (x1 x2 x3 x4 x5 : R) : List Poly → Prop
  | [] => True
  | f :: fs => evalPoly x1 x2 x3 x4 x5 f = 0 ∧ AllZero x1 x2 x3 x4 x5 fs

/-- `as[0] * fs[0] + as[1] * fs[1] + …`: a combination of the equations. -/
def comboZip : List Poly → List Poly → Poly
  | a :: as, f :: fs => addPoly (mulPoly a f) (comboZip as fs)
  | _, _ => []

theorem eval_comboZip : ∀ (as fs : List Poly), AllZero x1 x2 x3 x4 x5 fs →
    evalPoly x1 x2 x3 x4 x5 (comboZip as fs) = 0 := by
  intro as
  induction as with
  | nil => intro fs _; simp only [comboZip, evalPoly_nil]
  | cons a as ih =>
    intro fs hf
    match fs with
    | [] => simp only [comboZip, evalPoly_nil]
    | f :: fs => simp only [comboZip, eval_addPoly, eval_mulPoly, hf.1, ih fs hf.2]; grind

/-- A polynomial equal to a combination of `fs` vanishes wherever all of `fs` do. The
cofactors `as` are untrusted: a wrong one makes `hid` fail to close. -/
theorem zero_of_combo {p : Poly} (as fs : List Poly)
    (hid : subPoly p (comboZip as fs) = []) (hf : AllZero x1 x2 x3 x4 x5 fs) :
    evalPoly x1 x2 x3 x4 x5 p = 0 := by
  rw [eval_eq_of_subPoly_nil x1 x2 x3 x4 x5 hid, eval_comboZip x1 x2 x3 x4 x5 as fs hf]

end Eval

end PolySolve
