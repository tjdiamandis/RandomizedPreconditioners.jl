abstract type Sketch{T} end
abstract type FactoredSketch{T} <: Sketch{T} end


# ------------------------------------------------------------------------------
# |                               Nystrom Sketch                               |
# ------------------------------------------------------------------------------
struct NystromSketch{T} <: FactoredSketch{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

# Constructs Â_nys in factored form
# Â_nys = (AΩ)(ΩᵀAΩ)^†(AΩ)^ᵀ = UΛUᵀ
# [Martinsson & Tropp, Algorithm 16]
function NystromSketch(A::Matrix{T}, k::Int, r::Int; check=false) where {T <: Real}
    check && check_psd(A)
    n = size(A, 1)

    Ω = randn(n, r)
    Ω .= Array(qr(Ω).Q)
    Y = A * Ω
    ν = sqrt(n)*eps(norm(Y))                    #TODO: revisit this choice
    @. Y += ν * Ω
    B = Y / cholesky(Symmetric(Ω' * Y)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν)[1:k])

    return NystromSketch(U[:, 1:k], Λ)
end

# Define basic properties
Base.size(Anys::NystromSketch) = (size(Anys.U, 1), size(Anys.U, 1))
Base.size(Anys::NystromSketch, d::Int) = d <= 2 ? size(Anys)[d] : 1
LinearAlgebra.rank(Anys::NystromSketch) = size(Anys.U, 2)
Matrix(Anys::NystromSketch) = Anys.U*Anys.Λ*Anys.U'

# Define operations for Nystrom Sketch
LinearAlgebra.eigvals(Anys::NystromSketch) = Anys.Λ.diag

function LinearAlgebra.mul!(y, Anys::NystromSketch, x; cache=zeros(rank(Anys)))
    length(y) != length(x) || length(y) != size(Anys, 1) && error(DimensionMismatch())
    r = rank(Anys)
    @views mul!(cache[1:r], Anys.U', x)
    @views cache[1:r] .*= Anys.Λ.diag
    @views mul!(y, Anys.U, cache[1:r])
    return nothing
end

function LinearAlgebra.:*(Anys::NystromSketch, x::AbstractVector)
    n = size(Anys, 1)
    y = zeros(n)
    mul!(y, Anys, x)
    return y
end



# Increases rank until the approximation is sufficient
# By [Frangella et al., Prop 5.3], have that κ(P^{-1/2} * A * P^{-1/2}) ≤ (λᵣ + μ + ||E||)/μ
# TODO: Add verbose logging
# TODO: Could improve efficiency here, especially if the same sketch matrix is reused
function adaptive_nystrom_approx(A::Matrix{T}, r0::Int; r_inc_factor=2.0, k_factor=0.9, tol=1e-6, check=false, q=10) where {T <: Real}
    check && check_psd(A)
    n = size(A, 1)
    cache = (
        v0=zeros(n),
        v=zeros(n),
        Anys_mul=zeros(n)
    )
    r = r0
    Enorm = Inf
    Anys = nothing
    while Enorm > tol && r < n
        k = Int(round(k_factor*r))
        Anys = NystromSketch(A, k, r; check=check)
        Enorm = estimate_norm_E(A, Anys; q=q, cache=cache)
        r = Int(r_inc_factor*r)
    end
    return Anys
end

# Power method to estimate ||A - Anys||
function estimate_norm_E(A, Anys; q=10, cache=nothing)
    n = size(A, 1)
    if !isnothing(cache)
        v0, v = cache.v0, cache.v
    else
        v0, v = zeros(n), zeros(n)
        cache = (Anys_mul=zeros(n),)
    end
    
    v0 .= randn(n)
    normalize!(v0)

    Ehat = Inf
    for _ in 1:q
        mul!(v, Anys, v0; cache=cache.Anys_mul)
        mul!(v, A, v0, 1.0, -1.0)
        Ehat = dot(v0, v)
        normalize!(v)
        v0 .= v
    end
    return Ehat
end


# ------------------------------------------------------------------------------
# |                               Randomized SVD                               |
# ------------------------------------------------------------------------------
# TODO: