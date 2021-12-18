@testset "Nystrom Sketch" begin
    ## Compare "best" vs random preconditioner on random example
    # Data
    Random.seed!(0)
    n, r_true = 1000, 100
    A = randn(n, r_true)
    A = A*A'

    r = round(Int, r_true * 1.2)
    k = round(Int, 0.9*r)
    λ = eigvals(A; sortby=x->-x)
    Anys = RP.NystromSketch(A, k, r)
    @test λ[1:k] ≈ eigvals(Anys)
    @test ≈(RP.estimate_norm_E(A, Anys; q=10), opnorm(A - Matrix(Anys)); rtol=1e-2)

    k, r = k ÷ 2, r ÷ 2
    Anys = RP.NystromSketch(A, k, r)
    @test ≈(RP.estimate_norm_E(A, Anys; q=10), opnorm(A - Matrix(Anys)); rtol=1e-1)

    x = randn(n)
    y = Anys * x
    y ≈ Matrix(Anys) * x
    z = zeros(n)

    @testset "Adaptive Sketch Building" begin
        Anys_adapt = RP.adaptive_nystrom_approx(A, 2)
        @test round(128 - rank(Anys_adapt) * 1/.9) == 0       # Stops at r = 128, k = 115
        @test opnorm(A - Matrix(Anys_adapt)) <= 1e-6        
    end
end