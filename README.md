# Hip


[![Build Status](https://travis-ci.org/chriselrod/Hip.jl.svg?branch=master)](https://travis-ci.org/chriselrod/Hip.jl)

[![Coverage Status](https://coveralls.io/repos/chriselrod/Hip.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chriselrod/Hip.jl?branch=master)

[![codecov.io](http://codecov.io/github/chriselrod/Hip.jl/coverage.svg?branch=master)](http://codecov.io/github/chriselrod/Hip.jl?branch=master)




https://github.com/ROCm-Developer-Tools/HIP

Compile `hip_jl.cpp` with:
```
hipcc -O3 -shared -fPIC hip_jl.cpp -o hip_jl.so
```
and then it should work. Currently only `sgemm!` is a good idea.
With Vega graphics, `sgemm!` on 5000x5000 matrices takes CLBLAS 45 ms on my computer.
hipBLAS's sgemm:
```julia
julia> @benchmark sync_func($sgemm!, $hipC, 1f0, $hipA, $hipB, 0f0)
BenchmarkTools.Trial: 
  memory estimate:  288 bytes
  allocs estimate:  6
  --------------
  minimum time:     22.702 ms (0.00% GC)
  median time:      27.016 ms (0.00% GC)
  mean time:        26.965 ms (0.00% GC)
  maximum time:     27.429 ms (0.00% GC)
  --------------
  samples:          186
  evals/sample:     1
```

Because hipBLAS uses cuBLAS as a backend for NVidia cards, I'm sure the difference (vs CLBLAS) will be even more dramatic there.

Precompilation isn't currently supported, because I was wrapping the C++ functions via simply using `extern "C"`.
Before adding this as a dependency, I'll switch to either the usual `ccall` syntax of `ccall( (fuction, lib), ... )` or `CxxWrap`.

This is only really good if you want direct access to `HIP` functionality. Writing your own kernels, or even broadcast-type statements, currently means writing in `C++`. (Forking `transpiler` -- or better yet, making a sort of hipNative -- is beyond me for the forseeable future.)
For that reason, `CLArrays` and `CuArrays` are almost certainly much better choices unless you're desperate for BLAS performance on an AMD card.