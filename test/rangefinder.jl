@testset "Rangefinder" begin
    Random.seed!(0)
    m, n = 30, 20
    l = 5
    tol = 1e-8
    B = randn(m, l) * rand(l, n)

    Q = RP.rangefinder(B, 2l; q=0)
    @test norm(B - Q*Q'*B) <= tol

    Q = RP.rangefinder(B, 2l; q=5)
    @test norm(B - Q*Q'*B) <= tol

end