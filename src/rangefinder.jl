# [Martinsson & Tropp, Algorithm 7, 9]
function rangefinder(A::AbstractMatrix{T}, r::Int; q::Int=0, Ω=nothing, orthogonalize=true) where {T <: Number}
    m, n = size(A)
    Y = zeros(m, r)
    isnothing(Ω) && (Ω = GaussianTestMatrix(q > 0 ? m : n, r))

    if q == 0
        Z = nothing
    else
        Y .= Matrix(Ω)
        Z = zeros(n, r)
    end

    rangefinder!(Y, A, Ω; q=q, Z=Z, orthogonalize=orthogonalize)

    return Y
end

function rangefinder!(Y, A, Ω; q, Z, orthogonalize)
    if q == 0
        if Ω isa GaussianTestMatrix
            mul!(Y, A, Ω)
        else
            mul!(Y', Ω', A')
        end
    else
        # NOTE: with powering, cannot fundamentally accelerate the computation
        # start w Y = Ω
        for _ in 1:q
            Y .= Array(qr(Y).Q)
            # Y = A*A'*Y
            mul!(Z, A', Y)
            mul!(Y, A, Z)
        end
    end

    if orthogonalize
        Y .= Array(qr(Y).Q)
    end

    return nothing
end

function chebyshev_rangefinder(
    A::AbstractMatrix{T}, 
    ν::S, 
    r::Int; 
    q::Int=1, 
    Ω=nothing, 
    orthogonalize=true
) where {T <: Number, S <: AbstractFloat}
    m, n = size(A)

    isnothing(Ω) && (Ω = GaussianTestMatrix(m, r))
    Y = zeros(m, (q+1)*r)
    Z = zeros(n, r)
    
    Y0 = @view Y[:, 1:r]
    Y1 = @view Y[:, (r+1):(2*r)]

    Y0 .= Array(qr(Matrix(Ω)).Q)
    @views mul!(Z, A', Y0)
    @views mul!(Y1, A, Z)
    Y1 .*= (2 / ν)
    Y1 .-= Y0

    for i in 2:q
        Yi = @view Y[:, ((i)*r+1):((i+1)*r)]
        Yi1 = @view Y[:, ((i-1)*r+1):((i)*r)]
        Yi2 = @view Y[:, ((i-2)*r+1):((i-1)*r)]

        # Yᵢ = (4/ν)AAᵀYᵢ₋₁ - 2Yᵢ₋₁ - Yᵢ₋₂
        mul!(Z, A', Yi1)
        mul!(Yi, A, Z)
        Yi .*= (4 / ν)
        @. Yi -= 2Yi1 + Yi2
    end

    if orthogonalize
        Y .= Array(qr(Y).Q)
    end

    return Y

end

