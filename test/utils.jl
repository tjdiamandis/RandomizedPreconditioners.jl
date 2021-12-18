@testset "Utils" begin
    Random.seed!(0)
    n, r = 100, 10
    A = randn(n, r)
    A = A*A'

    # Check PSD
    @test RP.check_psd(A)
    @test_throws ArgumentError !RP.check_psd(A - 1e-1I)

    μ = 1e-2
    @test rank(A) - RP.deff(A, μ) < 2μ


end