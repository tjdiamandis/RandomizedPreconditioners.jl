abstract type Sketch{T} end
abstract type FactoredSketch{T} <: Sketch{T} end


# ------------------------------------------------------------------------------
# |                               Nystrom Sketch                               |
# ------------------------------------------------------------------------------
struct NystromSketch{T} <: FactoredSketch{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

#TODO: should this just wrap / generalize a NystromSketch (inc. flag for PSD)
# Specifically NystromSketch being a subtype of eigensketch
struct EigenSketch{T} <: FactoredSketch{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
end

# Constructs Â_nys in factored form
# Â_nys = (AΩ)(ΩᵀAΩ)^†(AΩ)^ᵀ = UΛUᵀ
# [Martinsson & Tropp, Algorithm 16]
function NystromSketch(A::AbstractMatrix{T}, k::Int, r::Int; check=false, q=0, Ω=nothing) where {T <: Real}
    check && check_psd(A)
    n = size(A, 1)
    Y = zeros(n, r)
    
    ν = sqrt(n)*eps(norm(A))                    #TODO: revisit this choice
    A[diagind(A)] .+= ν
    
    isnothing(Ω) && (Ω = GaussianTestMatrix(n, r))
    rangefinder!(Y, A, Ω; q=0, Z=nothing, orthogonalize=false)
    A[diagind(A)] .-= ν
    Z = zeros(r, r)
    mul!(Z, Ω', Y)

    B = Y / cholesky(Symmetric(Z)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν)[1:k])

    return NystromSketch(U[:, 1:k], Λ)
end

# When you want to skecth M = AᵀA
function NystromSketch_ATA(A::AbstractMatrix{T}, k::Int, r::Int) where {T}
    m, n = size(A)
    Y = zeros(n, r)
    cache = zeros(m, r)
    
    Ω = randn(n, r)
    mul!(cache, A, Ω)
    mul!(Y, A', cache)

    ν = sqrt(n)*eps(norm(Y))
    @. Y = Y + ν*Ω

    Z = zeros(r, r)
    mul!(Z, Ω', Y)

    B = Y / cholesky(Symmetric(Z)).U
    U, Σ, _ = svd(B)
    Λ = Diagonal(max.(0, Σ.^2 .- ν)[1:k])

    return NystromSketch(U[:, 1:k], Λ)
end


check_input(A, ::Type{NystromSketch}) = check_psd(A)
Sketch(A, k, r, ::Type{NystromSketch}; check=false, q=0) = NystromSketch(A, k, r; check=check)

# Define basic properties
Base.eltype(::Sketch{T}) where {T} = T
Base.size(Ahat::Union{NystromSketch, EigenSketch}) = (size(Ahat.U, 1), size(Ahat.U, 1))
Base.size(Ahat::FactoredSketch, d::Int) = d <= 2 ? size(Ahat)[d] : 1
LinearAlgebra.adjoint(A::Sketch) = Adjoint(A)
LinearAlgebra.rank(Ahat::FactoredSketch) = size(Ahat.U, 2)
LinearAlgebra.svdvals(Ahat::FactoredSketch) = Ahat.Λ.diag
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, Ahat::Union{NystromSketch, EigenSketch})
    Base.showarg(stdout, Ahat, true)
    println(io, "U factor:")
    show(io, mime, Ahat.U)
    println(io, "\nEigenvalues:")
    show(io, mime, Ahat.Λ.diag)
end
#TODO: this is a hack and definitely not best practice
function Base.show(io::IO, Ahat_::Adjoint{<:Any, <:Sketch})
    print(io, "Adjoint of ")
    show(io, parent(Ahat_))
end

# Define operations for Nystrom Sketch
Matrix(Anys::Union{NystromSketch, EigenSketch}) = Anys.U*Anys.Λ*Anys.U'
LinearAlgebra.eigvals(Anys::Union{NystromSketch, EigenSketch}) = Anys.Λ.diag    # decreasing order

function LinearAlgebra.mul!(y, Anys::Union{NystromSketch, EigenSketch}, x; cache=zeros(rank(Anys)))
    length(y) != length(x) || length(y) != size(Anys, 1) && error(DimensionMismatch())
    r = rank(Anys)
    @views mul!(cache[1:r], Anys.U', x)
    @views cache[1:r] .*= Anys.Λ.diag
    @views mul!(y, Anys.U, cache[1:r])
    return nothing
end

function LinearAlgebra.:*(Anys::Union{NystromSketch, EigenSketch}, x::AbstractVector)
    n = size(Anys, 1)
    y = zeros(n)
    mul!(y, Anys, x)
    return y
end


# ------------------------------------------------------------------------------
# |                               Randomized Eigendecomposition                               |
# ------------------------------------------------------------------------------
function EigenSketch(A::AbstractMatrix{T}, k::Int, r::Int; check=false, q::Int=0, Ω=nothing) where {T <: Real}
    check && !issymmetric(A) && throw(ArgumentError("A must be symmetric"))
    k > r && throw(ArgumentError("k must be less than r"))
    Q = rangefinder(A, r; q=q, Ω=Ω)
    C = Q' * A * Q
    @. C = 0.5 * (C + C')                       #TODO: is this needed for numerics?
    Λ, V = eigen(C; sortby=x->-real(x))
    V .= real.(V)
    Λ .= real(Λ)

    # Find largest magnitude eigenspace
    p = sortperm(Λ; by=x->-abs(x))
    Λ = Λ[p][1:k]
    V = V[:, p][:,1:k]

    # Return st eigvals sorted biggest to smallest
    pp = sortperm(Λ; by=x->-x)
    Λ .= Λ[pp]
    V .= V[:, pp]

    U = Q*V
    return EigenSketch(U[:, 1:k], Diagonal(Λ))
end

check_input(A, ::Type{EigenSketch}) = issymmetric(A)
Sketch(A, k, r, ::Type{EigenSketch}; check=false, q=5) = EigenSketch(A, k, r; check=check, q=q)


# ------------------------------------------------------------------------------
# |                               Randomized SVD                               |
# ------------------------------------------------------------------------------
struct RandomizedSVD{T} <: FactoredSketch{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
    V::Matrix{T}
end

# [Martinsson & Tropp, Algorithm 8]
# TODO: look at Tropp et al. 2019 & Halko et al. 2011 for more details on implementation
#   ‘Streaming low-rank matrix approximation with an application to scientiﬁc simulation’
#   'Finding structure with randomness: probabalistic algorithms for constructing 
#    approximate matrix decompositions'
function RandomizedSVD(A::AbstractMatrix{T}, k::Int, r::Int; q::Int=0, Ω=nothing) where {T <: Real}
    k > r && throw(ArgumentError("k must be less than r"))
    Q = rangefinder(A, r; q=q, Ω=Ω)
    C = Q'*A
    Û, Σ, V = svd(C)
    U = Q*Û
    return RandomizedSVD(U[:, 1:k], Diagonal(Σ[1:k]), V[:, 1:k])
end

check_input(A, ::Type{RandomizedSVD}) = nothing
Sketch(A, k, r, ::Type{RandomizedSVD}; check=false, q=5) = RandomizedSVD(A, k, r; q=q)

# Define basic operations
Base.size(Ahat::RandomizedSVD) = (size(Ahat.U, 1), size(Ahat.V, 1))
Matrix(Ahat::RandomizedSVD) = Ahat.U*Ahat.Λ*Ahat.V'
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, Ahat::RandomizedSVD)
    Base.showarg(stdout, Ahat, true)
    println(io, "\nU factor:")
    show(io, mime, Ahat.U)
    println(io, "\nsingular values:")
    show(io, mime, Ahat.Λ.diag)
    println(io, "\nV factor:")
    show(io, mime, Ahat.V)
end

function LinearAlgebra.mul!(y, Ahat::RandomizedSVD, x; cache=zeros(rank(Ahat)))
    length(x) != size(Ahat, 2) || length(y) != size(Ahat, 1) && error(DimensionMismatch())
    r = rank(Ahat)
    @views mul!(cache[1:r], Ahat.V', x)
    @views cache[1:r] .*= Ahat.Λ.diag
    @views mul!(y, Ahat.U, cache[1:r])
    return nothing
end

function LinearAlgebra.mul!(x, Ahat_::Adjoint{T, RandomizedSVD{T}}, y; cache=zeros(rank(Ahat))) where {T}
    Ahat = parent(Ahat_)
    length(x) != size(Ahat, 2) || length(y) != size(Ahat, 1) && error(DimensionMismatch())
    r = rank(Ahat)
    @views mul!(cache[1:r], Ahat.U', y)
    @views cache[1:r] .*= Ahat.Λ.diag
    @views mul!(x, Ahat.V, cache[1:r])
    return nothing
end

function LinearAlgebra.:*(Ahat::Union{Adjoint{T, RandomizedSVD{T}}, RandomizedSVD{T}}, x::AbstractVector) where {T}
    y = zeros(size(Ahat, 1))
    mul!(y, Ahat, x)
    return y
end



# ------------------------------------------------------------------------------
# |                             General Utilities                              |
# ------------------------------------------------------------------------------
# Power method to estimate ||A - Ahat||
function estimate_norm_E(A::AbstractMatrix{T}, Ahat::Sketch{T}; q=10, cache=nothing) where {T <: Number}
    m, n = size(A)
    if !isnothing(cache)
        u, v = cache.u, cache.v
    else
        u, v = zeros(m), zeros(n)
        cache = (Ahat_mul=zeros(min(m, n)),)
    end
    
    u .= randn(m)
    normalize!(u)
    v .= randn(n)
    normalize!(v)
    
    Ehat = Inf
    for _ in 1:q
        # u = (A - Ahat)*v
        mul!(u, Ahat, v; cache=cache.Ahat_mul)
        mul!(u, A, v, 1.0, -1.0)
        normalize!(u)

        # v = (A - Ahat)ᵀ*v
        if typeof(Ahat) <: Union{NystromSketch, EigenSketch}
            v .= u
        else
            mul!(v, Ahat', u; cache=cache.Ahat_mul)
            mul!(v, A', u, 1.0, -1.0)
            normalize!(v)
        end

        Ehat = abs(dot(u, A, v) - dot(u, Ahat, v))
    end
    return Ehat
end

# Increases rank until the approximation is sufficient
# By [Frangella et al., Prop 5.3], have that κ(P^{-1/2} * A * P^{-1/2}) ≤ (λᵣ + μ + ||E||)/μ
# TODO: Add verbose logging
# TODO: Could improve efficiency here, especially if the same sketch matrix is reused
function adaptive_sketch(
    A::AbstractMatrix{T}, r0::Int, SketchType::Type{<:Sketch};
    condition_number=false,
    ρ=1e-4,
    r_inc_factor=2.0,
    k_factor=0.9,
    tol=1e-6,
    check=false,
    q_norm=20,
    q_sketch=5,
    verbose=false
) where {T <: Real}
    check && check_input(A, SketchType)
    m, n = size(A)
    cache = (
        u=zeros(m),
        v=zeros(n),
        Ahat_mul=zeros(n)
    )
    r = r0
    Ahat = nothing
    error_metric = Inf
    while error_metric > tol && r < n
        k = round(Int, k_factor*r)
        Ahat = Sketch(A, k, r, SketchType; check=check, q=q_sketch)
        if condition_number
            error_metric = (Ahat.Λ[end] + ρ) / ρ - 1
            verbose && @info "κ = $error_metric, r = $r"
        else
            error_metric = estimate_norm_E(A, Ahat; q=q_norm, cache=cache)
            verbose && @info "||E|| = $error_metric, r = $r"
        end
        r = round(Int, r_inc_factor*r)
    end
    return Ahat
end