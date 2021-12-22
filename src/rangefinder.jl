# [Martinsson & Tropp, Algorithm 7]
# TODO: testing
function rangefinder(A::AbstractMatrix{T}, r::Int; q::Int=0) where {T <: Number}
    m, n = size(A)
    if q == 0
        Ω = 1/sqrt(n) * randn(n, r)
        Y = A * Ω
        Array(qr(Ω).Q)
    end

    Y = 1/sqrt(m) * randn(m, r)
    for i in 1:q
        Y = Array(qr(Y).Q)
        Y = A*A'*Y
    end
    return Array(qr(Y).Q)
end
