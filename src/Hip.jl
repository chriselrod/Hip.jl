#__precompile__(false)

module Hip

using Compat

export  HipArray,
        sgemm!,
        naiveMul!,
        naivesgemm!,
        tilesgemm!,
        free!,
        synchronize,
        set_device

include("hip_interface.jl")


end # module
