function eigmax_power!(y1, y2, M::AbstractMatrix; q::Int=0, tol=1e-3)
    n = size(M, 1)
    T = eltype(M)
    iszero(q) && (q = ceil(Int, 20*log(n)))

    randn!(y1)
    randn!(y2)
    normalize!(y1)
    normalize!(y2)
    ξ1, ξ2 = zero(T), zero(T)

    for _ in 1:q
        y1 .= y2

        mul!(y2, M, y1)
        ξ2 = dot(y1, y2)
        normalize!(y2)
        abs(ξ2 - ξ1) ≤ tol * ξ1 && break
    end
    return ξ2
end

function eigmax_power(M::AbstractMatrix; q::Int=0, tol=1e-3)
    n = size(M, 1)
    T = eltype(M)
    y1, y2 = zeros(T, n), zeros(T, n)
    return eigmax_power!(y1, y2, M; q=q, tol=tol)
end


function init_cache_eigmax_lanczos(n, max_iters)
    return (
        p = zeros(max_iters),
        w = zeros(max_iters),
        V = zeros(n, max_iters+1),
        tmp1 = Vector{Float64}(undef, n),
        tmp2 = Vector{Float64}(undef, n)
    )
end


function eig_lanczos(M::AbstractMatrix; q::Int=0, cache=nothing, tol=1e-12, eigtype=1) where {T}
    n = size(M, 1)
    iszero(q) && (q = ceil(Int, 20*log(n)))
    max_iters = min(q, n-1)

    if isnothing(cache)
        cache = init_cache_eigmax_lanczos(n, max_iters)
    end

    #initialize
    @views randn!(cache.V[:,1])
    @views cache.V[:,1] .= cache.V[:,1] ./ norm(cache.V[:,1])

    i = 1
    while i <= max_iters
        vi = @view(cache.V[:,i])
        vi1 = @view(cache.V[:,i+1])

        # quad form
        mul!(vi1, M, vi)
        cache.w[i] = dot(vi, vi1)

        vi1 .-= cache.w[i] .* vi
        if i > 1
            @views vi1 .-= cache.p[i-1] .* cache.V[:,i-1]
        end

        cache.p[i] = norm(vi1)
        cache.p[i] < tol && (break; i+=1)

        vi1 ./= cache.p[i]
        i += 1

    end
    i -= 1

    # LAPACK call for SymTridiagonal eigenvectors
    # https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.stegr!
    ind = eigtype == 1 ? i : 1
    return LAPACK.stegr!('N', 'I', cache.w[1:i], cache.p[1:i-1], 0.0, 0.0, ind, ind)[1][1]
end

eigmax_lanczos(M::AbstractMatrix; q::Int=0, cache=nothing, tol=1e-12) =
    eig_lanczos(M; q=q, cache=cache, tol=tol, eigtype=1)
eigmin_lanczos(M::AbstractMatrix; q::Int=0, cache=nothing, tol=1e-12) =
    eig_lanczos(M; q=q, cache=cache, tol=tol, eigtype=-1)
