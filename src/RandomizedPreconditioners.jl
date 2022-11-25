module RandomizedPreconditioners

using LinearAlgebra
using Random
using FFTW: dct!

include("utils.jl")
include("testmatrix.jl")
include("rangefinder.jl")
include("sketch.jl")
include("preconditioner.jl")
include("eig.jl")

export NystromSketch, NystromSketch_ATA, EigenSketch, RandomizedSVD, adaptive_sketch
export NystromPreconditioner, NystromPreconditionerInverse

end
