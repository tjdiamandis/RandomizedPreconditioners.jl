# RandomizedPreconditioners

[![Build Status](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl)

This package contains several _randomized preconditioners_, which use 
randomized numerical linear algebra to construct approximate inverses of matrices.

## Preconditioners

### Randomized Nyström Preconditioner [1]
Given a positive semidefinite matrix `A`, the Nyström Sketch `Â ≈ A` is constructed by
```julia
import RandomizedPreconditioners
const RP = RandomizedPreconditioners
Â = NystromSketch(A, k, r)
```
where `k` and `r` are parameters.

We can use `Â` to construct a preconditioner `P ≈ A + μ*I` for the system 
`(A + μ*I)x = b`, which is solved by conjugate gradients.

If you need `P` (e.g., `IterativeSolvers.jl`), use
```julia
P = RP.NystromPreconditioner(Anys, μ)
```

If you need `P⁻¹` (e.g., `Krylov.jl`), use
```julia
Pinv = RP.NystromPreconditionerInverse(Anys, μ)
```

These preconditioners can be simply passed into the solvers, for example
```julia
using Krylov
x, stats = cg(A+μ*I, b; M=Pinv)
```


## References
[1] Zachary Frangella, Joel A Tropp, and Madeleine Udell. “Randomized Nyström Precon-ditioning.” In:arXiv preprint arXiv:2110.02820(2021).