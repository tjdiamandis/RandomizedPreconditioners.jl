@testset "eigenvalues" begin
    Random.seed!(0)
    n, r_true = 500, 100
    A = randn(n, r_true)
    A = A*A'
    λ = eigmax(A)

    λ_power =  RP.eigmax_power(A)
    @test isapprox(λ_power, λ, rtol=1e-1)

    λmax_lanczos = RP.eigmax_lanczos(A)
    @test isapprox(λmax_lanczos, λ, rtol=1e-2)
    
    λmin_lanczos = RP.eigmin_lanczos(A + 1e-1*I)
    @test isapprox(λmin_lanczos, 1e-1, rtol=1e-2)
end