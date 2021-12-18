@testset "Nystrom Preconditioner" begin
    Random.seed!(0)
    n, r_true = 200, 20
    A = randn(n, r_true)
    A = A*A'
    
    r = round(Int, r_true * 1.2)
    k = round(Int, 0.9*r)
    μ = 1e-3
    Anys = RP.NystromSketch(A, k, r)
    
    D, V = eigen(A; sortby=x->-x)
    Vk = V[:,1:k]
    Dk = D[1:k]
    
    Ptrue = Symmetric(1/(D[k+1] + μ) * Vk*(Diagonal(Dk) + μ*I)*Vk' + (I - Vk*Vk'))
    Ptrue_inv = Symmetric((D[k+1] + μ) * Vk * Diagonal(1 ./ (Dk .+ μ)) * Vk' + (I - Vk*Vk'))

    
    @testset "Standard" begin
        P = RP.NystromPreconditioner(Anys, μ)
        @test eltype(P) == Float64
        @test opnorm(Ptrue - Matrix(P)) < μ
        for _ in 1:10
            x = randn(n)
            y = P \ x
            @test Ptrue \ x ≈ y
            ldiv!(P, x)
            @test y ≈ x
        end
        
        
    end
    
    @testset "Inverse" begin
        Pinv = RP.NystromPreconditionerInverse(Anys, μ)
        @test eltype(Pinv) == Float64
        @test opnorm(Ptrue_inv - Matrix(Pinv)) < μ
        for _ in 1:10
            x = randn(n)
            @test Ptrue_inv*x ≈ Pinv*x
        end
    end

end