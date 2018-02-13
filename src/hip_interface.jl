

using Compat

const hippath = Base.@__DIR__;
const lib_hip_jl = Libdl.dlopen(hippath * "/hip_jl.so")

const hipTranspose = Libdl.dlsym(lib_hip_jl, :tranpose)
const hipAlloc = Libdl.dlsym(lib_hip_jl, :hipArray)
const hipAllocUninitialized = Libdl.dlsym(lib_hip_jl, :hipArrayUninit)
const hipFree = Libdl.dlsym(lib_hip_jl, :hipDelete)
const hipDeviceToHost = Libdl.dlsym(lib_hip_jl, :hipMemDevToHost)
const sync = Libdl.dlsym(lib_hip_jl, :synchronize)
const setDevice = Libdl.dlsym(lib_hip_jl, :setDevice)
const sgemmNaive = Libdl.dlsym(lib_hip_jl, :naive_sgemm)
const sgemmTile = Libdl.dlsym(lib_hip_jl, :tiled_sgemm)
const sgemmHip = Libdl.dlsym(lib_hip_jl, :hipSgemm)


struct HipArray{N} <: AbstractArray{Float32,N}
    ptr::Ptr{Void}
    size::NTuple{N,Int}
    length::Int
end
function HipArray(x::AbstractArray{T,N}) where {T,N}
    HipArray(Array(Float32.(x)))
end
function HipArray(x::Array{Float32,N}) where {N}
    y = ccall(hipAlloc, Ptr{Cvoid}, (Ref{Cfloat}, Cint), x, Cint(length(x)))
    HipArray{N}(y, size(x), length(x))
end
function HipArray(x::NTuple{N,T}) where {N, T<: Integer}
    l = prod(x)
    y = ccall(hipAllocUninitialized, Ptr{Cvoid}, (Cint,), Cint(l))
    HipArray{N}(y, x, l)
end

function transpose!(x::HipArray{2}, y::HipArray{2})
    @assert y.size[1] == x.size[2]
    @assert y.size[2] == x.size[1]
    ccall(hipTranspose, Cvoid, (Ref{Cvoid}, Ref{Cvoid}, Cint, Cint), x.ptr, y.ptr, Cint(y.size[1]), Cint(y.size[2]))
end

function sgemm!(C::HipArray{2}, alpha::Cfloat, A::HipArray{2}, B::HipArray{2},
                    beta::Cfloat, transA = false, transB = false)
    @assert A.size[2] == B.size[1]
    @assert C.size[1] == A.size[1]
    @assert C.size[2] == B.size[2]
    ccall(sgemmHip, Cvoid, (Bool, Bool, Cint, Cint, Cint, Cfloat, Ref{Cvoid}, Ref{Cvoid}, Cfloat, Ref{Cvoid}), transA, transB, Cint(size(A, 1)), Cint(size(B,2)), Cint(size(A,2)), alpha, A.ptr, B.ptr, beta, C.ptr)
end

function naiveMul!(C::HipArray{2}, A::HipArray{2}, B::HipArray{2})
    @assert A.size[2] == B.size[1]
    @assert C.size[1] == A.size[1]
    @assert C.size[2] == B.size[2]
    ccall(sgemmNaive, Cvoid, (Cint, Cint, Cint, Cfloat, Ref{Cvoid}, Ref{Cvoid}, Cfloat, Ref{Cvoid}),
        Cint(size(A, 1)), Cint(size(B,2)), Cint(size(A,2)), 1f0, A.ptr, B.ptr, 0f0, C.ptr)
end
function naivesgemm!(C::HipArray{2}, alpha::Cfloat, A::HipArray{2}, B::HipArray{2}, beta::Cfloat)
    @assert A.size[2] == B.size[1]
    @assert C.size[1] == A.size[1]
    @assert C.size[2] == B.size[2]
    ccall(sgemmNaive, Cvoid, (Cint, Cint, Cint, Cfloat, Ref{Cvoid}, Ref{Cvoid}, Cfloat, Ref{Cvoid}),
        Cint(size(A, 1)), Cint(size(B,2)), Cint(size(A,2)), alpha, A.ptr, B.ptr, beta, C.ptr)
end
function tilesgemm!(C::HipArray{2}, alpha::Cfloat, A::HipArray{2}, B::HipArray{2}, beta::Cfloat)
    @assert A.size[2] == B.size[1]
    @assert C.size[1] == A.size[1]
    @assert C.size[2] == B.size[2]
    ccall(sgemmTile, Cvoid, (Cint, Cint, Cint, Cfloat, Ref{Cvoid}, Ref{Cvoid}, Cfloat, Ref{Cvoid}),
        Cint(size(A, 1)), Cint(size(B,2)), Cint(size(A,2)), alpha, A.ptr, B.ptr, beta, C.ptr)
end

Base.size(y::HipArray) = y.size
Base.length(y::HipArray) = y.length
Base.show(io::IO, y::HipArray) = println(io, "HipArray of size: $(y.size).")
Base.getindex(y::HipArray, i) = throw("Get index not defined for HipArray.")
Base.setindex!(y::HipArray, i, v) = throw("Set index not defined for HipArray.")


synchronize() = ccall(sync, Cvoid, ())
function set_device(i)
    ccall(setDevice, Cvoid, (Cint,), Cint(i))
end

function free!(y::HipArray)
    ccall(hipFree, Cvoid, (Ref{Cvoid},), y.ptr)
end
function Base.copy!(x::Array{Float32}, y::HipArray)#dims do not have to match.
    @assert length(x) == length(y)#but lengths do.
    ccall(hipDeviceToHost, Cvoid, (Ref{Cfloat}, Ref{Cvoid}, Cint), x, y.ptr, Cint(length(y)))
end
function Base.Array(y::HipArray{N}) where N
    out = Array{Float32,N}(uninitialized, y.size...)
    ccall(hipDeviceToHost, Cvoid, (Ref{Cfloat}, Ref{Cvoid}, Cint), out, y.ptr, Cint(length(y)))
    out
end


