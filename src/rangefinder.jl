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