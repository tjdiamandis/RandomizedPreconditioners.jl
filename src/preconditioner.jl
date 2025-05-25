abstract type Preconditioner{T} end

# ------------------------------------------------------------------------------
# |                           Nystrom Preconditioner                           |
# ------------------------------------------------------------------------------
mutable struct NystromPreconditioner{T <: Real, V <: AbstractVector{T}} <: Preconditioner{T}
    A_nys::NystromSketch{T}
    λ::T
    μ::T
    cache::V
    function NystromPreconditioner(A_nys::NystromSketch{T,AT,V}, μ::T) where {T <: Real, AT, V}
        A_nys_Λ_diag_end = first(collect(A_nys.Λ.diag[end:end]))  # bring to CPU if needed
        cache = V(undef, rank(A_nys))
        fill!(cache, zero(T))
        return new{T,V}(A_nys, A_nys_Λ_diag_end, μ, cache)
    end
end
function Matrix(P::NystromPreconditioner)
    return Symmetric(
        1/(P.λ + P.μ) * P.A_nys.U*(P.A_nys.Λ + P.μ*I)*P.A_nys.U' 
        + (I - P.A_nys.U*P.A_nys.U')
    )
end

# We care about applying P⁻¹
# P⁻¹x = U*(λ + μ)*(Λ + μI)⁻¹*Uᵀ*x + (I - UUᵀ)*x
#      = x - U*((λ + μ)*(Λ + μI)⁻¹ + I)*Uᵀ*x
function LinearAlgebra.ldiv!(y::AbstractVector{T}, P::NystromPreconditioner{T}, x::AbstractVector{T}) where {T <: Real}
    length(y) != length(x) && error(DimensionMismatch())
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) * 1 / (P.A_nys.Λ.diag + P.μ) - 1
    mul!(y, P.A_nys.U, P.cache)
    @. y = x + y
    return nothing
end

function LinearAlgebra.ldiv!(P::NystromPreconditioner{T}, x::AbstractVector{T}) where {T <: Real} 
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) / (P.A_nys.Λ.diag + P.μ) - one(T)
    x .+= P.A_nys.U*P.cache
    return nothing
end

function LinearAlgebra.:\(P::NystromPreconditioner{T}, x::AbstractVector{T}) where {T <: Real} 
    n = size(P, 1)
    y = similar(x, n)
    ldiv!(y, P, x)
    return y
end


# Used for Krylov.jl
struct NystromPreconditionerInverse{T <: Real, V <: AbstractVector{T}}
    A_nys::NystromSketch{T}
    λ::T
    μ::T
    cache::V
    function NystromPreconditionerInverse(A_nys::NystromSketch{T,AT,V}, μ::T) where {T <: Real, AT, V}
        A_nys_Λ_diag_end = first(collect(A_nys.Λ.diag[end:end]))  # bring to CPU if needed
        cache = V(undef, rank(A_nys))
        fill!(cache, zero(T))
        return new{T,V}(A_nys, A_nys_Λ_diag_end, μ, cache)
    end
end
Base.size(P::Union{NystromPreconditionerInverse, NystromPreconditioner}) = (size(P.A_nys.U, 1), size(P.A_nys.U, 1))
Base.size(P::Union{NystromPreconditionerInverse, NystromPreconditioner}, d::Int) = d <= 2 ? size(P)[d] : 1
Base.eltype(::Union{NystromPreconditionerInverse{T}, NystromPreconditioner{T}}) where {T} = T
function Matrix(P::NystromPreconditionerInverse)
    return Symmetric(
        (P.λ + P.μ) * P.A_nys.U*inv(P.A_nys.Λ + P.μ*I)*P.A_nys.U' 
        + (I - P.A_nys.U*P.A_nys.U')
    )
end

function LinearAlgebra.mul!(y, P::NystromPreconditionerInverse{T}, x::AbstractVector{T}) where {T <: Real}
    length(y) != length(x) && error(DimensionMismatch())
    
    mul!(P.cache, P.A_nys.U', x)
    @. P.cache *= (P.λ + P.μ) / (P.A_nys.Λ.diag + P.μ) - one(T)
    mul!(y, P.A_nys.U, P.cache)
    @. y = x + y
end

function LinearAlgebra.:*(P::NystromPreconditionerInverse{T}, x::AbstractVector{T}) where {T <: Real}
    y = similar(x)
    mul!(y, P, x)
    return y
end