using Hip
using Base.Test





using Compat, BenchmarkTools
const hippath = Pkg.dir("Hip") * "/src";
const lib_hip_jl = Libdl.dlopen(hippath * "/hip_jl.so")


# write your own tests here
const N = 4096
a = randn(Float32, N, N);
b = randn(Float32, N, N);
c = zeros(a);

ah = HipArray(a);
bh = HipArray(b);
#ch = HipArray(c);
cht = HipArray(c);

tilesgemm!(cht, 1f0, ah, bh, 0f0)
A_mul_B!(c, a, b)
synchronize()
cres = Array(cht)


chh = HipArray(c);
sgemm!(chh, 1f0, ah, bh, 0f0)
synchronize()
cres2 = Array(chh)




free!(ah)
free!(bh)
free!(cht)
N2 = 32
a = randn(Float32, N2, N2);
b = randn(Float32, N2, N2);
c = zeros(a);
ah = HipArray(a);
bh = HipArray(b);
ch = HipArray(c);
tilesgemm!(ch, 1f0, ah, bh, 0f0)
A_mul_B!(c, a, b)
synchronize(); Array(ch)

A_mul_Bt!(c, a, b)
At_mul_B!(c, a, b)
At_mul_Bt!(c, a, b)
Array(ch)

function set_ab!(a, b, ia = 0.25f0, ja = 1f0, ib = 0.5f0, jb = 2f0)
    for i in 1:32
        for j in 1:32
            a[j,i] = ia*i + ja*j
            b[j,i] = ib*i + jb*j
        end
    end
end
free!(ah)
free!(bh)
ah = HipArray(a);
bh = HipArray(b);
tilesgemm!(ch, 1f0, ah, bh, 0f0)
A_mul_B!(c, a, b)
copy!(cha, ch); cha



chn = HipArray(c);
naivesgemm!(chn, 1f0, ah, bh, 0f0)


chh = HipArray(c);
sgemm!(chh, 1f0, ah, bh, 0f0)

function sync_func(f, c, alpha, a, b, beta) 
    f(c, alpha, a, b, beta)
    synchronize()
end

copy!(cres, chn); cres
copy!(cres, chh); cres

@benchmark A_mul_B!($c, $a, $b)
@benchmark sync_func($naivesgemm!, $cht, 1f0, $ah, $bh, 0f0)
@benchmark sync_func($tilesgemm!, $cht, 1f0, $ah, $bh, 0f0)
@benchmark sync_func($sgemm!, $chh, 1f0, $ah, $bh, 0f0)



const THREADS_PER_BLOCK_X = 32
const THREADS_PER_BLOCK_Y = 32
const hipBlockDim_x = 32
const hipBlockDim_y = 32
const hipBlockIdx_x = 0
const hipBlockIdx_y = 0

function run_kern!(c, alpha, a, b, beta)

    M, K = size(a)
    N = size(b,2)

    c .*= beta

    num_tiles = div(K,THREADS_PER_BLOCK_X);

    tileA = Vector{Float32}(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
    tileB = Vector{Float32}(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y)
    for t in 0:num_tiles-1


        for i in 0:THREADS_PER_BLOCK_X-1, j in 0:THREADS_PER_BLOCK_Y-1
            kern_shared!(tileA, tileB, i, j, t, M, K, a, b)
        end
        for i in 0:THREADS_PER_BLOCK_X-1, j in 0:THREADS_PER_BLOCK_Y-1
            kern_tile!(c, a, b, i, j, alpha, tileA, tileB, M)
        end


    end
    c, tileA, tileB
end
function kern_shared!(tileA, tileB, hipThreadIdx_x, hipThreadIdx_y, t, M, K, A, B)
    idx_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    idx_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    id_local = hipThreadIdx_y + THREADS_PER_BLOCK_Y * hipThreadIdx_x;
    id_localT = hipThreadIdx_x + THREADS_PER_BLOCK_X * hipThreadIdx_y;

    t_col = t*THREADS_PER_BLOCK_X;
    tileA[id_localT+1] = A[(t_col + hipThreadIdx_x)*M + idx_y+1];
    tileB[id_local+1] = B[t_col + hipThreadIdx_y + K*idx_x+1];
end

function kern_tile!(C, A, B, hipThreadIdx_x, hipThreadIdx_y, alpha, tileA, tileB, M)

    idx_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    idx_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    myIdx = idx_y + idx_x * M;


    local_c = zero(eltype(C))

    idxA = hipThreadIdx_y * THREADS_PER_BLOCK_X ;
    idxB = hipThreadIdx_x * THREADS_PER_BLOCK_X ;

    for k in 0:THREADS_PER_BLOCK_X-1
        local_c += tileA[k + idxA+1] * tileB[k + idxB+1];
    end


    C[myIdx+1] += alpha * local_c;
end


