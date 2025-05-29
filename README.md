# RandomizedPreconditioners

[![Build Status](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tjdiamandis/RandomizedPreconditioners.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/tjdiamandis/RandomizedPreconditioners.jl)

This package contains several _randomized preconditioners_, which use 
randomized numerical linear algebra to construct approximate inverses of matrices.
These approximate inverses can dramatically speed up iterative linear system solvers.

## Preconditioners

### Positive Definite Systems: Randomized Nyström Preconditioner [1]
Construct a randomized Nyström preconditioner `P ≈ A + μ*I` for a positive definite linear system `Ax = b` with
```julia
using RandomizedPreconditioners
P = NystromPreconditioner(A)
```
If you need `P⁻¹` use
```julia
Pinv = NystromPreconditionerInverse(A)
```
The preconditioner has the form `P ≈ Â + μ*I`, where `Â` an Nyström sketch of `A` and `μ` is a small positive number. The default value of `μ` is `1e-6`, but it can be specified by passing `μ` to the constructor:
```julia
P = NystromPreconditioner(A; μ=1e-3)
```

These preconditioners can be simply passed into the solvers, for example
```julia
using Krylov

# Use NystromPreconditionerInverse
x, stats = cg(A+μ*I, b; M=Pinv)

# Use NystromPreconditioner
x, stats = cg(A+μ*I, b; M=P, ldiv=true)
```
Pay attention to if your solver uses `P` (and applies the preconditioner with `ldiv!`) or `P⁻¹` (and applies the preconditioner with `mul!`).

The package [`LinearSolve.jl`](https://github.com/SciML/LinearSolve.jl) defines
a convenient common interface to access all the Krylov implementations, which
makes testing very easy.
```julia
using RandomizedPreconditioners, LinearSolve
Â = NystromSketch(A, k, r)
P = NystromPreconditioner(Â, μ)

prob = LinearProblem(A, b)
sol = solve(prob, IterativeSolversJL_CG(), Pl=P)
```


## Sketches
The sketching algorithms below use a rank `r` sketching matrix and have complexity
`O(n²r)`. The parameter `k ≤ r` truncates the sketch, which can improve numerical
performance. Possible choices include `k = r - 10` and `k = round(Int, 0.95*r)`.
Sketches allow for faster (approximate) multiplication (`*` and `mul!`) and are
used to construct preconditioners.

If passing a matrix-free operator A, you can set the type of the cache as a keyword argument, 
`NystromSketch(A, r::Int; n=nothing, S=Array{Float64,2})`.

### Positive Semidefinite Matrices: Nyström Sketch [2, Alg. 16]
Given a positive semidefinite matrix `A`, construct the Nyström Sketch `Â ≈ A` via
```julia
using RandomizedPreconditioners
Â = NystromSketch(A)
```

The sketch size `r` and the preconditioner size `k` (where `k ≤ r`) may also be specified directly:
```julia
Â = NystromSketch(A; k=k, r=r)
```

### Symmetric Matrices: Eigen Sketch / Generalized Nyström Sketch
```julia
using RandomizedPreconditioners
Â = EigenSketch(A, k, r)
```

### General Matrices: Randomized SVD [2, Alg. 8] 
The Randomized SVD uses the powered randomized rangefinder [2, Alg. 9] with
powering parameter `q`. Small values of `q` (e.g., `5`) seem to perform 
well. Note that the complexity increases to `O(n²rq)`.
```julia
using RandomizedPreconditioners
Â = RandomizedSVD(A, k, r; q=10)
```

## Eigenvalues
We implement two algorithms for a randomized estimate of the maximum eigenvalue
for a PSD matrix: the power method and the Lanczos method.
```julia
using RandomizedPreconditioners
const RP = RandomizedPreconditioners

λmax_power =  RP.eigmax_power(A)
λmax_lanczos = RP.eigmax_lanczos(A)
λmin_lanczos = RP.eigmin_lanczos(A)
```
The Lanczos method can estimate the maximum and minimum eigenvalue simultaneously:
```julia 
λmax, λmin = RP.eig_lanczos(A; eigtype=0)
```

## Test Matrices
There are several choices for the random embedding used in the algorithms.
By default, this package uses Gaussian embeddings (and Gaussian test matrices),
but it also includes the `SSFT` and the ability to add new test matrices by
implementing the `TestMatrix` interface.

A `TestMatrix`, `Ω`, should implement matrix multiplication for itself and its
adjoint by implementing the `!mul` method. 
See Martinsson and Tropp [2] Section 9 for more.

## Roadmap
- Test Matrices
    - [X] TestMatrix type
    - [ ] Sparse maps
    - [X] Subsampled trigonometric transform
    - [ ] DCT & Hartley option for SSRFT
    - [ ] Tensor product maps
- Rangefinders
    - [ ] Lanzcos randomized rangefinder
    - [X] Chebyshev randomized rangefinder
    - [ ] Incremental rangefinder with updating
    - [ ] Subsequent orthogonalization
    - [ ] A posteriori error estimation
    - [ ] Incremental rangefinder with powering
    - [ ] Incremental rangefinder for sparse matrices
- Sketches & Factorizations
    - [ ] Powering option / incorporating rangefinder into Nystrom sketch 
    - [ ] powerURV (w. re-orthonormalization)
    - [ ] CPQR decomposition
    - [ ] Improve randomized SVD
- Preconditioners
    - [ ] Add preconditioner for symmetric systems
    - [ ] Preconditioner for least squares
- Performance
    - [ ] Avoid redoing computations in adaptive sketch
    - [ ] General performance
- Documentation
    - [ ] More complete general docs
    - [ ] Least squares example (sketch & solve, iterative sketching, sketch & precondition)


## References
[1] Zachary Frangella, Joel A Tropp, and Madeleine Udell. “Randomized Nyström Preconditioning.” In:arXiv preprint arXiv:2110.02820(2021). https://arxiv.org/abs/2110.02820

[2] PG Martinsson and JA Tropp. “Randomized numerical linear algebra: foundations & algorithms (2020).” In: arXiv preprint arXiv:2002.01387. https://arxiv.org/abs/2002.01387
