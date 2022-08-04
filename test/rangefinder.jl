@testset "Rangefinder" begin
    Random.seed!(0)
    m, n = 30, 20
    l = 5
    tol = 1e-8
    B = randn(m, l) * rand(l, n)

    @testset "Standard" begin
        Q = RP.rangefinder(B, 2l; q=0)
        @test norm(B - Q*Q'*B) <= tol

        Q = RP.rangefinder(B, 2l; q=5)
        @test norm(B - Q*Q'*B) <= tol

        Ω = RP.GaussianTestMatrix(n, 2l)
        Q = RP.rangefinder(B, 2l; q=0, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol

        Ω = RP.GaussianTestMatrix(m, 2l)
        Q = RP.rangefinder(B, 2l; q=5, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol

        Ω = RP.SSFTTestMatrix(n, 2l)
        Q = RP.rangefinder(B, 2l; q=0, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol

        Ω = RP.SSFTTestMatrix(m, 2l)
        Q = RP.rangefinder(B, 2l; q=5, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol
    end

    @testset "Chebyshev" begin
        ν = 2 * sqrt(RP.eigmax_power(B'*B))

        r = 5
        Q = RP.chebyshev_rangefinder(B, ν, r; q=1)
        @test norm(B - Q*Q'*B) <= tol

        r = 2
        Q = RP.chebyshev_rangefinder(B, ν, r; q=5)
        @test norm(B - Q*Q'*B) <= tol

        r = 5
        Ω = RP.GaussianTestMatrix(m, r)
        Q = RP.chebyshev_rangefinder(B, ν, r; q=1, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol

        r = 2
        Ω = RP.GaussianTestMatrix(m, r)
        Q = RP.chebyshev_rangefinder(B, ν, r; q=5, Ω=Ω)
        @test norm(B - Q*Q'*B) <= tol
    end

end