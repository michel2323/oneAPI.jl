# oneMKL FFT (DFT) high-level Julia interface
# Inspired by AMDGPU ROCFFT interface style, adapted to oneMKL DFT C wrapper.

module FFT

using ..oneMKL
using ..oneMKL: oneAPI, SYCL, syclQueue_t
using ..Support
using ..SYCL
using LinearAlgebra
using GPUArrays
using AbstractFFTs
import AbstractFFTs: complexfloat, realfloat
import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!
import AbstractFFTs: plan_rfft, plan_brfft, plan_inv, normalization
import AbstractFFTs: fft, bfft, ifft, rfft, Plan, ScaledPlan
export MKLFFTPlan

# Low-level enums mirroring C API (subset)
# (We can just re-use integer constants; C wrappers return 0 on success.)
const DFT_PREC_SINGLE = 0
const DFT_PREC_DOUBLE = 1
const DFT_DOM_REAL    = 0
const DFT_DOM_COMPLEX = 1

# Placement values
const DFT_PARAM_DIMENSION = 1
const DFT_PARAM_LENGTHS   = 2
const DFT_PARAM_PRECISION = 3
const DFT_PARAM_FORWARD_SCALE = 4
const DFT_PARAM_BACKWARD_SCALE = 5

# Opaque descriptor type alias to Ptr{Nothing} (generated wrapper not yet exposed)
# We'll declare ccall prototypes manually until generator exposes them.

# NOTE: The liboneapi_support.jl generated file currently doesn't have DFT entries; add manual ccalls.
const lib = :liboneapi_support

# Allow implicit conversion of SYCL queue object to raw handle when storing/passing
Base.convert(::Type{syclQueue_t}, q::SYCL.syclQueue) = Base.unsafe_convert(syclQueue_t, q)

# Creation / destruction
ccall_create1d(desc_ref, prec::Int32, dom::Int32, length::Int64) = ccall((:onemklDftCreate1D, lib), Cint, (Ref{Ptr{Cvoid}}, Cint, Cint, Int64), desc_ref, prec, dom, length)
ccall_creatend(desc_ref, prec::Int32, dom::Int32, dim::Int64, lengths::Ptr{Int64}) = ccall((:onemklDftCreateND, lib), Cint, (Ref{Ptr{Cvoid}}, Cint, Cint, Int64, Ptr{Int64}), desc_ref, prec, dom, dim, lengths)
ccall_destroy(desc) = ccall((:onemklDftDestroy, lib), Cint, (Ptr{Cvoid},), desc)
ccall_commit(desc, q) = ccall((:onemklDftCommit, lib), Cint, (Ptr{Cvoid}, syclQueue_t), desc, q)
ccall_fwd(desc, ptr) = ccall((:onemklDftComputeForward, lib), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), desc, ptr)
ccall_fwd_oop(desc, pin, pout) = ccall((:onemklDftComputeForwardOutOfPlace, lib), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), desc, pin, pout)
ccall_bwd(desc, ptr) = ccall((:onemklDftComputeBackward, lib), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), desc, ptr)
ccall_bwd_oop(desc, pin, pout) = ccall((:onemklDftComputeBackwardOutOfPlace, lib), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), desc, pin, pout)
ccall_set_double(desc, param::Int32, value::Float64) = ccall((:onemklDftSetValueDouble, lib), Cint, (Ptr{Cvoid}, Cint, Float64), desc, param, value)

abstract type MKLFFTPlan{T,K,inplace} <: AbstractFFTs.Plan{T} end

Base.eltype(::MKLFFTPlan{T}) where T = T
is_inplace(::MKLFFTPlan{<:Any,<:Any,inplace}) where inplace = inplace

# Forward / inverse flags
const MKLFFT_FORWARD = true
const MKLFFT_INVERSE = false

mutable struct cMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::Ptr{Cvoid}
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    realdomain::Bool
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
end

# Real transforms use separate struct (mirroring AMDGPU style) for buffer staging
mutable struct rMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::Ptr{Cvoid}
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    xtype::Symbol
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
end

# Inverse plan constructors (derive from existing plan)
function plan_inv(p::cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p)
    p.pinv = q
    q
end
function plan_inv(p::cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p)
    p.pinv = q
    q
end

function plan_inv(p::rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    # forward real -> inverse complex->real (brfft)
    q = rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:brfft,p.region,p.buffer,p)
    p.pinv = q
    q
end
function plan_inv(p::rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    # inverse real -> forward real (rfft)
    q = rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:rfft,p.region,p.buffer,p)
    p.pinv = q
    q
end

function Base.show(io::IO, p::MKLFFTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "oneMKL FFT in-place " : "oneMKL FFT ", K ? "forward" : "inverse", " plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T)
end

# Plan constructors
function _create_descriptor(sz::NTuple{N,Int}, T::Type, complex::Bool; normalize=true) where N
    prec = T<:Float64 || T<:ComplexF64 ? DFT_PREC_DOUBLE : DFT_PREC_SINGLE
    dom = complex ? DFT_DOM_COMPLEX : DFT_DOM_REAL
    desc_ref = Ref{Ptr{Cvoid}}()
    lengths = collect(Int64, sz)
    iprec = Int32(prec); idom = Int32(dom)
    st = length(lengths) == 1 ? ccall_create1d(desc_ref, iprec, idom, lengths[1]) : ccall_creatend(desc_ref, iprec, idom, length(lengths), pointer(lengths))
    st == 0 || error("onemkl DFT create failed (status $st)")
    desc = desc_ref[]
    # Set scaling so that forward is unscaled, inverse multiplies by 1/volume (AbstractFFTs convention)
    if normalize
        vol = prod(sz)
        stfs = ccall_set_double(desc, Int32(DFT_PARAM_FORWARD_SCALE), 1.0)
        stfs == 0 || error("set forward scale failed ($stfs)")
        stbs = ccall_set_double(desc, Int32(DFT_PARAM_BACKWARD_SCALE), 1.0/vol)
        stbs == 0 || error("set backward scale failed ($stbs)")
    end
    # Construct a SYCL queue from current Level Zero context/device (reuse global queue)
    ze_ctx = oneAPI.context(); ze_dev = oneAPI.device()
    sycl_dev = SYCL.syclDevice(SYCL.syclPlatform(oneAPI.driver()), ze_dev)
    sycl_ctx = SYCL.syclContext([sycl_dev], ze_ctx)
    q = SYCL.syclQueue(sycl_ctx, sycl_dev, oneAPI.global_queue(ze_ctx, ze_dev))
    stc = ccall_commit(desc, q)
    stc == 0 || error("onemkl DFT commit failed (status $stc)")
    return desc, q
end

# Complex plans
function plan_fft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    desc, q = _create_descriptor(size(X), T, true)
    return cMKLFFTPlan{T,MKLFFT_FORWARD,false,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end
function plan_bfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    desc, q = _create_descriptor(size(X), T, true)
    return cMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end

# In-place (provide separate methods)
function plan_fft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    desc,q = _create_descriptor(size(X),T,true)
    cMKLFFTPlan{T,MKLFFT_FORWARD,true,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end
function plan_bfft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    desc,q = _create_descriptor(size(X),T,true)
    cMKLFFTPlan{T,MKLFFT_INVERSE,true,N,R,Nothing}(desc,q,size(X),size(X),false,reg,nothing,nothing)
end

# Real forward (out-of-place)
function plan_rfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    desc,q = _create_descriptor(size(X),T,false)
    xdims = size(X)
    # output last region dim becomes N/2+1 (assume first region index)
    ax = reg[1]
    ydims = Base.setindex(xdims, div(xdims[ax],2)+1, ax)
    buffer = oneAPI.oneArray{Complex{T}}(undef, ydims)
    rMKLFFTPlan{T,MKLFFT_FORWARD,false,N,R,typeof(buffer)}(desc,q,xdims,ydims,:rfft,reg,buffer,nothing)
end

# Real inverse (complex->real) requires complex input shape
function plan_brfft(X::oneAPI.oneArray{T,N}, d::Integer, region) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    # output real size 'd' along region[1]
    xdims = size(X)
    ydims = Base.setindex(xdims, d, reg[1])
    # Extract underlying real type R from Complex{R}
    @assert T <: Complex
    RT = T.parameters[1]
    desc,q = _create_descriptor(ydims, RT, false)
    buffer = oneAPI.oneArray{T}(undef, xdims) # copy for safety
    rMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,typeof(buffer)}(desc,q,xdims,ydims,:brfft,reg,buffer,nothing)
end

# Convenience no-region methods use all dimensions in order
plan_fft(X::oneAPI.oneArray) = plan_fft(X, ntuple(identity, ndims(X)))
plan_bfft(X::oneAPI.oneArray) = plan_bfft(X, ntuple(identity, ndims(X)))
plan_fft!(X::oneAPI.oneArray) = plan_fft!(X, ntuple(identity, ndims(X)))
plan_bfft!(X::oneAPI.oneArray) = plan_bfft!(X, ntuple(identity, ndims(X)))
plan_rfft(X::oneAPI.oneArray) = plan_rfft(X, (1,))  # default first dim like Base.rfft
plan_brfft(X::oneAPI.oneArray, d::Integer) = plan_brfft(X, d, (1,))

# Alias names to mirror AMDGPU / AbstractFFTs style
const plan_ifft = plan_bfft
const plan_ifft! = plan_bfft!
const plan_irfft = plan_brfft

# Inversion
Base.inv(p::MKLFFTPlan) = plan_inv(p)

# High-level wrappers operating like CPU FFTW versions.
function fft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft(X) * X)
end
function ifft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_bfft(X) * X)
end
function fft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft!(X) * X; X)
end
function ifft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_bfft!(X) * X; X)
end
function rfft(X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    (plan_rfft(X) * X)
end
function irfft(X::oneAPI.oneArray{T}, d::Integer) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_brfft(X, d) * X)
end

# Execution helpers
_rawptr(a::oneAPI.oneArray{T}) where T = reinterpret(Ptr{Cvoid}, pointer(a))

function _exec!(p::cMKLFFTPlan{T,MKLFFT_FORWARD,true}, X::oneAPI.oneArray{T}) where T
    st = ccall_fwd(p.handle, _rawptr(X)); st==0 || error("forward FFT failed ($st)"); X
end
function _exec!(p::cMKLFFTPlan{T,MKLFFT_INVERSE,true}, X::oneAPI.oneArray{T}) where T
    st = ccall_bwd(p.handle, _rawptr(X)); st==0 || error("inverse FFT failed ($st)"); X
end
function _exec!(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{T}) where {T,K}
    st = (K==MKLFFT_FORWARD ? ccall_fwd_oop : ccall_bwd_oop)(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("FFT failed ($st)"); Y
end

# Real forward
function _exec!(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{Complex{T}}) where T
    st = ccall_fwd_oop(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("rfft failed ($st)"); Y
end
# Real inverse (complex -> real)
function _exec!(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{R}) where {R,T<:Complex{R}}
    st = ccall_bwd_oop(p.handle, _rawptr(X), _rawptr(Y)); st==0 || error("brfft failed ($st)"); Y
end

# Public API similar to AMDGPU
function Base.:*(p::cMKLFFTPlan{T,K,true}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X)
end
function Base.:*(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    Y = oneAPI.oneArray{T}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{T}, p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X,Y)
end

# Real forward
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    Y = oneAPI.oneArray{Complex{T}}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{Complex{T}}, p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    _exec!(p,X,Y)
end
# Real inverse
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    Y = oneAPI.oneArray{R}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{R}, p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    _exec!(p,X,Y)
end

end # module FFT
