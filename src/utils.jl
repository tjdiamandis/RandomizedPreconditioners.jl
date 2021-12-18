# theoretically, for NystromSketch choose r = 2⌈1.5deff(µ)⌉ + 1.
function deff(A::AbstractMatrix, μ; check=false)
    check && check_psd(A)
    λ = eigvals(A)
    return sum(x->x/(x+μ), λ)
end

function check_psd(A::AbstractMatrix)
    n = size(A, 1)
    psd_tol = sqrt(n)*eps(norm(A))
    !isposdef(A + psd_tol*I) && throw(ArgumentError("A must be PSD"))
    return true
end
