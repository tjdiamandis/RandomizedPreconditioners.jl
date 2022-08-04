# [Martinsson & Tropp, Algorithm 7, 9]
function rangefinder(A::AbstractMatrix{T}, r::Int; q::Int=0, Ω=nothing, orthogonalize=true) where {T<:Number}
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

function chebyshev_rangefinder(A::AbstractMatrix{T}, ν::Float64, r::Int; q::Int=1, Ω=nothing, orthogonalize=true) where {T<:Number}
    m, n = size(A)

    isnothing(Ω) && (Ω = GaussianTestMatrix(m, r))

    Y0 = qr(Ω.Ω).Q
    Y1 = (2 / ν) * A * A' * Y0 - Y0
    Y = hcat(Y0, Y1)

    Yi_2 = Y0
    Yi_1 = Y1
    for i in 2:q
        Yi = (4 / ν) * A * A' * Yi_1 - 2 * Yi_2 - Yi_2
        Y = hcat(Y, Yi)
        Yi_2 = Yi_1
        Yi_1 = Yi
    end

    return Array(qr(Y).Q)

end

export rangefinder, chebyshev_rangefinder