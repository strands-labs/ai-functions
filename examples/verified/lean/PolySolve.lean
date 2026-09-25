import PolySolve.Elimination

/-!
# How many solutions does a polynomial system have?

A solver, msolve, counts the solutions of a system, but too slowly for the
system asked about. Eliminating `x5` first gives a smaller system it can finish.
The answer is then trusted and proved in two parts:

```text
trusted:  msolve's count for the system it was actually given
proved:   that system is the original with x5 eliminated (EliminatesX5)
assumed:  eliminating x5 keeps the count (CountsSurviveElimination, in the contract)
```

Counting solutions needs the complex numbers and finite sets, which this theory
does not have without Mathlib. So `HasSolutionCount` is opaque: msolve's guarantee
is its only source, and the step from `EliminatesX5` to equal counts is a
hypothesis of the contract rather than a theorem.
-/

namespace PolySolve

/-- `system`, read in the variables `x1` … `xk`, has exactly `n` distinct complex
solutions. It must not mention a variable after `xk`. -/
opaque HasSolutionCount (k : Nat) (system : List RawPoly) (n : Nat) : Prop

/-- The number of solutions msolve reports for `system` in `x1` … `xk`. Looked up with
a tool, which also vouches for `HasSolutionCount k system` of that number. -/
opaque msolveCount (k : Nat) (system : List RawPoly) : Nat

/-- Eliminating `x5` keeps the number of solutions. It holds because `EliminatesX5`
makes projection a bijection between the solution sets over `ℂ`; the contract takes
it as a hypothesis, since this theory cannot state it as a theorem. -/
def CountsSurviveElimination : Prop :=
  ∀ fs gs n, EliminatesX5 fs gs → HasSolutionCount 4 gs n → HasSolutionCount 5 fs n

/-- **Contract.** `answer` is the number of solutions of `system` in `x1` … `x5`. -/
def CountCorrect (answer : Nat) (system : List RawPoly) : Prop :=
  CountsSurviveElimination → HasSolutionCount 5 system answer

/-- The route through an elimination: a passing certificate for `gs`, and a count for
`gs`, give the count for `fs`. -/
theorem countCorrect_of_elimination (fs gs : List RawPoly) (c : Certificate)
    (hc : c.check (polysOfRaw fs) (polysOfRaw gs) = true) {n : Nat} (hn : HasSolutionCount 4 gs n) :
    CountCorrect n fs :=
  fun survive => survive fs gs n (eliminatesX5_of_check fs gs c hc) hn

/-- The direct route: a count for the system itself. -/
theorem countCorrect_of_count {fs : List RawPoly} {n : Nat} (hn : HasSolutionCount 5 fs n) :
    CountCorrect n fs :=
  fun _ => hn

end PolySolve
