using Random, LinearAlgebra, Krylov
using Plots
using RandomizedPreconditioners
const RP = RandomizedPreconditioners

n = 500
μ = 1e-4

# Build Spectrum (exp decay)
λs = max.(√n * exp.(-(0:n-1) ./ √n), μ)
plot(λs, lw=3, legend=false, title="Spectrum")

# Build Matrix
V = Array(qr(randn(n,n)).Q)
A = V*Diagonal(λs)*V'
A = 0.5(A + A')

# Build system
xtrue = randn(n)
b = A * xtrue

# Build preconditioner
r = 110
k = round(Int, r - 5)
Anys = NystromSketch(A, k, r)
P = RP.NystromPreconditionerInverse(Anys, μ)

# Solve system
_, stats_npc = cg(A, b; history=true)
_, stats_nys = cg(A, b; history=true, M = P)
_, stats_diag = cg(A, b; history=true, M=Diagonal(diag(A)))

nb = norm(b)
res_npc = prepend!(stats_npc.residuals, nb)
res_nys = prepend!(stats_nys.residuals, nb)
res_diag = prepend!(stats_diag.residuals, nb)

plt_cg = plot(
    res_npc,
    dpi=300,
    lw=2,
    label="No Preconditioner",
    ylabel="residual",
    xlabel="iteration",
    title="Convergence of CG",
    legend=:topright,
    yaxis=:log
)
plot!(plt_cg, 
    res_nys,
    label="Nystrom Preconditioner",
    lw=2
)
plot!(plt_cg, 
    res_diag,
    label="Diagonal Preconditioner",
    lw=2
)
display(plt_cg)