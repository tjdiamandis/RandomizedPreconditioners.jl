# RandomizedPreconditioners

[![Build Status](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl)

This package contains several _randomized preconditioners_, which use 
randomized numerical linear algebra to construct approximate inverses of matrices.
These approximate inverses can dramatically speed up iterative linear system solvers.

## Preconditioners

### Positive Definite Systems: Randomized Nyström Preconditioner [1]
Given a positive semidefinite matrix `A`, the Nyström Sketch `Â ≈ A` is constructed by
```julia
import RandomizedPreconditioners
const RP = RandomizedPreconditioners
Â = RP.NystromSketch(A, k, r)
```
where `k` and `r` are parameters with `k ≤ r`.

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


## Sketches
The sketching algorithms below use a rank `r` sketching matrix and have complexity
`O(n²r)`. The parameter `k ≤ r` truncates the sketch, which can improve numerical
performance. Possible choices include `k = r - 10` and `k = round(Int, 0.95*r)`.
Sketches allow for faster (approximate) multiplication (`*` and `mul!`) and are
used to construct preconditioners.

### Positive Semidefinite Matrices: Nyström Sketch [2, Alg. 16]
```julia
import RandomizedPreconditioners
const RP = RandomizedPreconditioners
Â = RP.NystromSketch(A, k, r)
```

### General Matrices: Randomized SVD [2, Alg. 8] 
The Randomized SVD uses the powered randomized rangefinder [2, Alg. 9] with
powering parameter `q`. Small values of `q` (e.g., `5`) seem to perform 
well. Note that the complexity increases to `O(n²rq)`.
```julia
import RandomizedPreconditioners
const RP = RandomizedPreconditioners
Â = RP.RandomizedSVD(A, k, r; q=10)
```


## References
[1] Zachary Frangella, Joel A Tropp, and Madeleine Udell. “Randomized Nyström Preconditioning.” In:arXiv preprint arXiv:2110.02820(2021). https://arxiv.org/abs/2110.02820

[2] PG Martinsson and JA Tropp. “Randomized numerical linear algebra: foundations & algorithms (2020).” In: arXiv preprint arXiv:2002.01387. https://arxiv.org/abs/2002.01387
