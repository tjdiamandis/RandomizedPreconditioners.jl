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

check_input(A, ::Type{NystromSketch}) = check_psd(A)
Sketch(A, k, r, ::Type{NystromSketch}; check=false, q=0) = NystromSketch(A, k, r; check=check)

# Define basic properties
Base.eltype(::Sketch{T}) where {T} = T
Base.size(Ahat::NystromSketch) = (size(Ahat.U, 1), size(Ahat.U, 1))
Base.size(Ahat::FactoredSketch, d::Int) = d <= 2 ? size(Ahat)[d] : 1
LinearAlgebra.adjoint(A::Sketch) = Adjoint(A)
LinearAlgebra.rank(Ahat::FactoredSketch) = size(Ahat.U, 2)
LinearAlgebra.svdvals(Ahat::FactoredSketch) = Ahat.Λ.diag
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, Ahat::NystromSketch)
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
Matrix(Anys::NystromSketch) = Anys.U*Anys.Λ*Anys.U'
LinearAlgebra.eigvals(Anys::NystromSketch) = Anys.Λ.diag    # decreasing order

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



# ------------------------------------------------------------------------------
# |                               Randomized SVD                               |
# ------------------------------------------------------------------------------
#TODO: switch Vt for V?
struct RandomizedSVD{T} <: FactoredSketch{T}
    U::Matrix{T}
    Λ::Diagonal{T, Vector{T}}
    Vt::Matrix{T}
end

# [Martinsson & Tropp, Algorithm 8]
# TODO: look at Tropp et al. 2019 & Halko et al. 2011 for more details on implementation
#   ‘Streaming low-rank matrix approximation with an application to scientiﬁc simulation’
#   'Finding structure with randomness: probabalistic algorithms for constructing 
#    approximate matrix decompositions'
function RandomizedSVD(A::Matrix{T}, k::Int, r::Int; q::Int=0) where {T <: Real}
    k > r && throw(ArgumentError("k must be less than r"))
    Q = rangefinder(A, r; q=q)
    C = Q'*A
    Û, Σ, V = svd(C)
    U = Q*Û
    return RandomizedSVD(U[:, 1:k], Diagonal(Σ[1:k]), V'[1:k, :])
end

check_input(A, ::Type{RandomizedSVD}) = nothing
Sketch(A, k, r, ::Type{RandomizedSVD}; check=false, q=5) = RandomizedSVD(A, k, r; q=q)

# Define basic operations
Base.size(Ahat::RandomizedSVD) = (size(Ahat.U, 1), size(Ahat.Vt, 2))
Matrix(Ahat::RandomizedSVD) = Ahat.U*Ahat.Λ*Ahat.Vt
function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, Ahat::RandomizedSVD)
    Base.showarg(stdout, Ahat, true)
    println(io, "\nU factor:")
    show(io, mime, Ahat.U)
    println(io, "\nsingular values:")
    show(io, mime, Ahat.Λ.diag)
    println(io, "\nV factor:")
    show(io, mime, Ahat.Vt)
end

function LinearAlgebra.mul!(y, Ahat::RandomizedSVD, x; cache=zeros(rank(Ahat)))
    length(x) != size(Ahat, 2) || length(y) != size(Ahat, 1) && error(DimensionMismatch())
    r = rank(Ahat)
    @views mul!(cache[1:r], Ahat.Vt, x)
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
    @views mul!(x, Ahat.Vt', cache[1:r])
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
        if Ahat isa NystromSketch
            v .= u
        else
            mul!(v, Ahat', u; cache=cache.Ahat_mul)
            mul!(v, A', u, 1.0, -1.0)
            normalize!(v)
        end

        Ehat = dot(u, A, v) - dot(u, Ahat, v)
    end
    return Ehat
end

# Increases rank until the approximation is sufficient
# By [Frangella et al., Prop 5.3], have that κ(P^{-1/2} * A * P^{-1/2}) ≤ (λᵣ + μ + ||E||)/μ
# TODO: Add verbose logging
# TODO: Could improve efficiency here, especially if the same sketch matrix is reused
function adaptive_approx(A::Matrix{T}, r0::Int, SketchType::Type{<:Sketch}; r_inc_factor=2.0, k_factor=0.9, tol=1e-6, check=false, q_norm=20, q_sketch=5, verbose=false) where {T <: Real}
    check && check_input(A, SketchType)
    m, n = size(A)
    cache = (
        u=zeros(m),
        v=zeros(n),
        Ahat_mul=zeros(n)
    )
    r = r0
    Enorm = Inf
    Ahat = nothing
    while Enorm > tol && r < n
        k = round(Int, k_factor*r)
        Ahat = Sketch(A, k, r, SketchType; check=check, q=q_sketch)
        Enorm = estimate_norm_E(A, Ahat; q=q_norm, cache=cache)
        verbose && @info "||E|| = $Enorm, r = $r"
        r = round(Int, r_inc_factor*r)
    end
    return Ahat
end