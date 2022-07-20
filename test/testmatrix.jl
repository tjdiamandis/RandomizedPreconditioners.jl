@testset "TestMatrix" begin
    Random.seed!(0)
    n, r_true = 100, 10
    A = randn(n, r_true)
    A = A*A'

    S = RP.GaussianTestMatrix(n, 2*r_true; orthonormal=true)
    Y = similar(S.Ω)
    mul!(Y', S', A')
    @test A * Matrix(S) ≈ Y

    Z = similar(S.Ω')
    mul!(Z, S', A)
    @test Matrix(S)' * A ≈ Z

    S = RP.SSFTTestMatrix(n, 2*r_true)
    mul!(Y', S', A')
    @test A * Matrix(S) ≈ Y

    mul!(Z, S', A)
    @test Matrix(S)' * A ≈ Z
end