using FFTW

# We highly recommend Gaussians and random partial isometries (Section 8). 
# Sparse maps, SRTTs, and tensor random embeddings (Section 9) also work very well. 
# In practice, all these approaches exhibit similar behavior; 
# see Section 11.5 for more discussion. 
# We present analysis only for Gaussian dimension reduction because it is both simple and precise.

# Sparse Maps -- input matrix is Sparse
# SRTTs - when fast trig availible
# Tensor product maps

# TestMatrix type for randomized algorithms
# - convention is that Testmatrix Ω in n × k, where n > k
# - TestMatrix should implement:
#   1) mul: M → Ω*M i.e., mul!(Y, Ω, M) for matrix M
#   2) mul for adjoint: M → Ωᵀ*M i.e., mul!(Y, Ω', M) for matrix M
abstract type TestMatrix end

struct GaussianTestMatrix{T} <: TestMatrix
    Ω::Matrix{T}        # n x k
end

function GaussianTestMatrix(n, r; orthonormal=false)
    Ω = 1/sqrt(n) * randn(n, r)
    orthonormal && (Ω .= Array(qr(Ω).Q))
    return GaussianTestMatrix(Ω)
end
Base.size(S::GaussianTestMatrix) = size(S.Ω)
LinearAlgebra.adjoint(S::TestMatrix) = Adjoint(S)

LinearAlgebra.mul!(Y, S::GaussianTestMatrix, X) = mul!(Y, S.Ω, X)
LinearAlgebra.mul!(Y, Sadj::Adjoint{<:Any, <:GaussianTestMatrix}, X) = mul!(Y, Sadj.parent.Ω', X)


# Assumption: data is real
struct SSFTTestMatrix{T} <: TestMatrix
    p1::Vector{Int}     # perm 1
    p2::Vector{Int}     # perm 2
    r::Vector{Int}      # restriction
    e1::Vector{Int}     # sign flip 1
    e2::Vector{Int}     # sign flip 2
    cache::Vector{T}    # cache for mul
end

function SSFTTestMatrix(n, k; T=Float64)
    p1 = randperm(n)
    p2 = randperm(n)
    e1 = rand((-1,1), n)
    e2 = rand((-1,1), n)
    r = randperm(n)[1:k]
    cache = zeros(T, n)
    return SSFTTestMatrix(p1, p2, r, e1, e2, cache)
end
Base.size(S::SSFTTestMatrix) = (S.n, S.k)
Base.eltype(::SSFTTestMatrix{T}) where {T} = T

function LinearAlgebra.mul!(Y, Sadj::Adjoint{<:Any, <:SSFTTestMatrix{T}}, X) where {T}
    S = Sadj.parent
    n = length(S.p1)
    d = length(S.r)
    
    for i in 1:n
        S.cache .= X[S.p1, i]
        S.cache .*= S.e1
        dct!(S.cache)
        @. Y[:,i] = sqrt(n/d) * S.cache[S.r]
    end
    return nothing
end

function LinearAlgebra.mul!(Y, S::SSFTTestMatrix{T}, X) where {T}
    n = length(S.p1)
    d = length(S.r)

    for i in 1:n
        Yi = @view Y[:,i]
        Yi .= zero(T)
        @views Yi[S.r] .= X[:, i]
        idct!(Yi)
        Yi .*= S.e1
        S.cache[S.p1] .= Yi
        Yi .*= sqrt(n/d)
    end
    return nothing
end