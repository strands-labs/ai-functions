import PolySolve.Raw

/-!
# Eliminating `x5`, with a certificate

`EliminatesX5 fs gs` says that `gs` is `fs` with `x5` eliminated: every solution
of `gs` extends to exactly one solution of `fs`. Projection is then a bijection
between the two solution sets, so the systems have the same number of solutions.

The agent proves it with a `Certificate`: three sets of cofactors, which are
untrusted data. `Certificate.check` recomputes every identity they claim, and
`eliminatesX5_of_check` turns a passing check into the statement. A wrong
cofactor makes the check `false`, never a wrong theorem.
-/

namespace PolySolve

open Lean.Grind

/-- No term mentions `x5`. -/
def noX5 (ps : List Poly) : Bool := ps.all fun p => p.all fun t => t.mono.e5 == 0

/-- The solutions of `gs`, which does not mention `x5`, are exactly the projections of
the solutions of `fs`, and each extends in only one way. This is claimed over every
commutative ring in which nonzero integers are invertible: `ℚ`, `ℂ`, and every other
field of characteristic 0. -/
def EliminatesX5 (fs gs : List RawPoly) : Prop :=
  noX5 (polysOfRaw gs) = true ∧
  ∀ {R : Type} [CommRing R] (inv : Int → R), (∀ n : Int, n ≠ 0 → (n : R) * inv n = 1) →
    ∀ x1 x2 x3 x4 : R,
      (AllZero x1 x2 x3 x4 0 (polysOfRaw gs) ↔ ∃ x5, AllZero x1 x2 x3 x4 x5 (polysOfRaw fs)) ∧
      ∀ a b : R, AllZero x1 x2 x3 x4 a (polysOfRaw fs) → AllZero x1 x2 x3 x4 b (polysOfRaw fs) → a = b

/-- The agent's evidence that `gs` eliminates `x5` from `fs`. -/
structure Certificate where
  /-- Forward: `gs[i] = Σ_j forward[i][j] * fs[j]`, so `gs` vanishes wherever `fs` does. -/
  forward : List (List Poly)
  /-- The recovery: at a solution of `gs`, `x5 = recover(x1, …, x4)`. -/
  recover : Poly
  /-- Backward: `scale * fs[k](x1, …, x4, recover) = Σ_j backward[k][j] * gs[j]`. -/
  backward : List (List Poly)
  scale : Int
  /-- Uniqueness: `unique_scale * (x5 - recover) = Σ_j unique[j] * fs[j]`. -/
  unique : List Poly
  unique_scale : Int

/-- `gs[i] - combination(as[i], fs) = 0` for every `i`, with matching lengths. -/
def combosMatch : List Poly → List (List Poly) → List Poly → Bool
  | [], [], _ => true
  | g :: gs, as :: ass, fs => subPoly g (comboZip as fs) == [] && combosMatch gs ass fs
  | _, _, _ => false

/-- `L * f(x1, …, x4, X5) - combination(bs[k], gs) = 0` for every `f`. -/
def substMatch (L : Int) (X5 : Poly) : List Poly → List (List Poly) → List Poly → Bool
  | [], [], _ => true
  | f :: fs, b :: bs, gs =>
      subPoly (scaleInt L (substVar f 5 X5)) (comboZip b gs) == [] && substMatch L X5 fs bs gs
  | _, _, _ => false

def Certificate.check (c : Certificate) (fs gs : List Poly) : Bool :=
  noX5 gs && noX5 [c.recover] && c.scale != 0 && c.unique_scale != 0 &&
  combosMatch gs c.forward fs && substMatch c.scale c.recover fs c.backward gs &&
  subPoly (scaleInt c.unique_scale (subPoly (var 5) c.recover)) (comboZip c.unique fs) == []

section Soundness

variable {R : Type} [CommRing R]

section Vars
variable (x1 x2 x3 x4 x5 : R)
@[simp] theorem eval_var1 : evalPoly x1 x2 x3 x4 x5 (var 1) = x1 := by simp only [var, evalPoly, evalMono]; grind
@[simp] theorem eval_var2 : evalPoly x1 x2 x3 x4 x5 (var 2) = x2 := by simp only [var, evalPoly, evalMono]; grind
@[simp] theorem eval_var3 : evalPoly x1 x2 x3 x4 x5 (var 3) = x3 := by simp only [var, evalPoly, evalMono]; grind
@[simp] theorem eval_var4 : evalPoly x1 x2 x3 x4 x5 (var 4) = x4 := by simp only [var, evalPoly, evalMono]; grind
@[simp] theorem eval_var5 : evalPoly x1 x2 x3 x4 x5 (var 5) = x5 := by simp only [var, evalPoly, evalMono]; grind
end Vars

theorem eval_substVar5 (x1 x2 x3 x4 x5 : R) (f q : Poly) :
    evalPoly x1 x2 x3 x4 x5 (substVar f 5 q) = evalPoly x1 x2 x3 x4 (evalPoly x1 x2 x3 x4 x5 q) f := by
  simp [substVar, eval_compose]

theorem eval_of_noX5 {p : Poly} (h : (p.all fun t => t.mono.e5 == 0) = true) (x1 x2 x3 x4 a b : R) :
    evalPoly x1 x2 x3 x4 a p = evalPoly x1 x2 x3 x4 b p := by
  induction p with
  | nil => rfl
  | cons t p ih =>
    simp only [List.all_cons, Bool.and_eq_true, beq_iff_eq] at h
    simp only [evalPoly_cons, evalMono, h.1, Semiring.pow_zero, ih h.2]

theorem allZero_of_noX5 {gs : List Poly} (h : noX5 gs = true) (x1 x2 x3 x4 a b : R) :
    AllZero x1 x2 x3 x4 a gs → AllZero x1 x2 x3 x4 b gs := by
  induction gs with
  | nil => intro; trivial
  | cons g gs ih =>
    simp only [noX5, List.all_cons, Bool.and_eq_true] at h
    intro ⟨hg, hgs⟩
    exact ⟨(eval_of_noX5 h.1 x1 x2 x3 x4 a b) ▸ hg, ih h.2 hgs⟩

theorem allZero_of_combosMatch (x1 x2 x3 x4 x5 : R) (fs : List Poly) (hf : AllZero x1 x2 x3 x4 x5 fs) :
    ∀ gs ass, combosMatch gs ass fs = true → AllZero x1 x2 x3 x4 x5 gs := by
  intro gs
  induction gs with
  | nil => intros; trivial
  | cons g gs ih =>
    intro ass h
    match ass with
    | [] => simp [combosMatch] at h
    | as :: ass =>
      simp only [combosMatch, Bool.and_eq_true, beq_iff_eq] at h
      exact ⟨zero_of_combo x1 x2 x3 x4 x5 as fs h.1 hf, ih ass h.2⟩

/-- `n * y = 0` with `n` invertible gives `y = 0`. -/
theorem eq_zero_of_mul (inv : Int → R) {n : Int} (hinv : (n : R) * inv n = 1) {y : R}
    (h : (n : R) * y = 0) : y = 0 := by
  have : y = inv n * ((n : R) * y) := by grind
  rw [this, h]; grind

theorem allZero_of_substMatch (inv : Int → R) (x1 x2 x3 x4 : R) {L : Int}
    (hinv : (L : R) * inv L = 1) (X5 : Poly) (gs : List Poly) (hg : AllZero x1 x2 x3 x4 0 gs) :
    ∀ fs bs, substMatch L X5 fs bs gs = true →
      AllZero x1 x2 x3 x4 (evalPoly x1 x2 x3 x4 0 X5) fs := by
  intro fs
  induction fs with
  | nil => intros; trivial
  | cons f fs ih =>
    intro bs h
    match bs with
    | [] => simp [substMatch] at h
    | b :: bs =>
      simp only [substMatch, Bool.and_eq_true, beq_iff_eq] at h
      refine ⟨?_, ih bs h.2⟩
      have hz := zero_of_combo x1 x2 x3 x4 0 b gs h.1 hg
      rw [eval_scaleInt, eval_substVar5] at hz
      exact eq_zero_of_mul inv hinv hz

theorem eliminatesX5_of_check (fs gs : List RawPoly) (c : Certificate)
    (h : c.check (polysOfRaw fs) (polysOfRaw gs) = true) : EliminatesX5 fs gs := by
  simp only [Certificate.check, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨hgv, hrv⟩, hL⟩, hM⟩, hfwd⟩, hbwd⟩, huniq⟩ := h
  refine ⟨hgv, fun {R} _ inv hinv x1 x2 x3 x4 => ⟨⟨fun hg => ?_, fun ⟨x5, hf⟩ => ?_⟩, fun a b ha hb => ?_⟩⟩
  · exact ⟨_, allZero_of_substMatch inv x1 x2 x3 x4 (hinv _ hL) c.recover _ hg _ _ hbwd⟩
  · exact allZero_of_noX5 hgv x1 x2 x3 x4 x5 0
      (allZero_of_combosMatch x1 x2 x3 x4 x5 _ hf _ _ hfwd)
  · -- At any solution, `x5 = recover(x1, …, x4)`, which does not depend on `x5`.
    have hr : (c.recover.all fun t => t.mono.e5 == 0) = true := by simpa [noX5] using hrv
    have at_ (x5 : R) (hf : AllZero x1 x2 x3 x4 x5 (polysOfRaw fs)) :
        x5 = evalPoly x1 x2 x3 x4 0 c.recover := by
      have hz := zero_of_combo x1 x2 x3 x4 x5 c.unique _ huniq hf
      rw [eval_scaleInt, eval_subPoly, eval_var5] at hz
      have := eq_zero_of_mul inv (hinv _ hM) hz
      rw [eval_of_noX5 hr x1 x2 x3 x4 x5 0] at this
      grind
    rw [at_ a ha, at_ b hb]

end Soundness

end PolySolve
