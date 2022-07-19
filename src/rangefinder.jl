# [Martinsson & Tropp, Algorithm 7]
function rangefinder(A::AbstractMatrix{T}, r::Int; q::Int=0, Ω=nothing, orthogonalize=true) where {T <: Number}
    m, n = size(A)
    Y = zeros(m, r)
    
    if q == 0
        isnothing(Ω) && (Ω = GaussianTestMatrix(n, r))
        Z = nothing
    else
        Y .= GaussianTestMatrix(m, r).Ω
        Z = zeros(n, r)
    end

    rangefinder!(Y, A, Ω; q=q, Z=Z, orthogonalize=orthogonalize)

    return Y
end

function rangefinder!(Y, A, Ω; q, Z, orthogonalize)
    if q == 0
        mul!(Y', Ω', A')
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