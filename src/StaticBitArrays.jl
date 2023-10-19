module StaticBitArrays

using StaticArrays
const SA = StaticArrays

export SBitVector, SBitMatrix, SBitArray

struct SBitArray{S,C,T<:Unsigned,N,L} <: StaticArray{S,Bool,N}
    # type signiture: {Length, ChunksLength, ChunksType}
    chunks::SVector{C,T}
    function SBitArray{S}(s::SVector{C,T}) where {S,C,T}
        S <: Tuple || error("type parameter S: $S must be a Tuple type of Int, e.g. `SBitArray{Tuple{4,5}}`")
        N = SA.tuple_length(S)
        L = SA.tuple_prod(S)
        length(s) == nchunks(T, L) || error("register/length mismatch")
        new{S,C,T,N,L}(s)
    end
    function SBitArray{S, C, T, N, L}(s::StaticArraysCore.StaticArray) where {S<:Tuple, C, T<:Unsigned, N, L}
        new{S,C,T,N,L}(s)
    end
end

const SBitVector{S1} = SBitArray{Tuple{S1}} where S1
const SBitMatrix{S1,S2} = SBitArray{Tuple{S1,S2}} where {S1,S2}

# Convenience constructors for an array of booleans, not type stable
SBitVector(a::NTuple{S,Bool}) where S = SBitVector{S}(UInt64, a)
SBitVector(::Type{U}, a::NTuple{S,Bool}) where {U,S} = SBitVector{S}(U, a)
SBitArray{S}(a::Union{AbstractArray{Bool},NTuple{<:Any,Bool}}) where S = SBitArray{S}(UInt64, a)
function SBitArray{S}(::Type{T}, a::Union{AbstractArray{Bool},NTuple{<:Any,Bool}}) where {S,T<:Unsigned}
    N = SA.tuple_length(S)
    L = SA.tuple_prod(S)
    C = nchunks(T, L)
    length(a) == L || error("Length of prod(S): $L does not match length of array a $(length(a))")
    SBitArray{S,C,T,N,L}(a)
end
# We need the union for method ambiguities with StaticArrays
function SBitArray{S,C,T,N,L}(
    a::Union{StaticArray{S,Bool,L},AbstractArray{Bool},NTuple{L,Bool}}
) where {S<:Tuple,C,T,N,L}
    nb = nbits(T)
    # build an ntuple for the number of complete chunks, C - 1
    chunks = ntuple(C - 1) do c
        # loop over bits to build the chunk
        chunk = zero(T)
        for b in 1:nbits(T)
            i = (c - 1) * nb + b
            chunk |= a[i] << (b - 1)
        end
        chunk
    end
    # Calculate the last chunk separately to avoid branching in the loop,
    # which slows down the constructor for AbstractArray inputs.
    lastchunk = zero(T)
    # Use a bit mask to get the length of the last chunk
    lastchunklen = (L - 1) & (nb - 1) + 1
    for b in 1:lastchunklen
        i = (C - 1) * nb + b
        lastchunk |= a[i] << (b - 1)
    end
    chunks = (chunks..., lastchunk)
    SBitArray{S}(SVector{C,T}(chunks))
end

struct BitComb{f,O,L,A}
    args::A
end
BitComb{f,O,L}(args::A) where {f,O,L,A} = BitComb{f,O,L,A}(args)

@generated function Base.map(F, a1::SBitArray{S,C,T,N,L}, as::StaticArray...) where {S,C,T,N,L}
    f = F.instance
    if Core.Compiler.return_type(f, Tuple{a1, tuple_contents(as)...}) <: Bool
        expr = Expr(:tuple)
        offset = 0
        for i in 1:C
            n = nbits(T)
            offset = (i - 1) * n
            l = min(n, L - offset)
            part = :(reduce(BitComb{$f,$offset,L}((a1, as...)), ntuple(identity, Val{$l}()); init=Int64(0)))

            exprs = Vector{Expr}(undef, prod(S))
            for i in 1:prod(S)
                tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
                exprs[i] = :(f($(tmp...)))
            end
            push!(expr.args, part)
        end

        return :(SBitArray{S,C,T,N,L}(SVector{C,T}($expr)))
    else
        return :(StaticArrays._map(f, a1, as...))
    end
end

@inline function (bc::BitComb{f,O,L})(acc, i) where {f,O,L}
    i1 = i + O
    @inbounds b = f(map(a -> a[i1], bc.args)...)::Bool
    return acc & b << (L - 1)
end

const NANDFUNCS = Union{typeof(!),typeof(nand)}
const BITWISE2 = Union{typeof(|),typeof(&),typeof(xor),typeof(nand)}

function Base.map(f::NANDFUNCS, a::SBitArray{S,C,T,N,L}) where {S,C,T,N,L}
    return SBitArray{S,C,T,N,L}(map(nand, a.chunks))
end
function Base.map(f::BITWISE2, a1::A, as::A...) where A<:SBitArray
    return map(f, parent(a1), map(parent(as))...)
end

Base.@assume_effects :foldable function tuple_contents(::Type{X}) where {X<:Tuple}
    return tuple(X.parameters...)
end
tuple_contents(xs::Tuple) = xs

Base.size(::SBitArray{S}) where {S} = S
Base.IndexStyle(::Type{<:SBitArray}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::SBitArray{<:Any,<:Any,T}, ind::Int) where T
    readbit(s.chunks[nchunks(T, ind)], ind)
end
function readbit(chunk::T, i::Int) where T
    shift = UInt(i - 1) & (UInt(nbits(T)) - one(UInt))
    mask = one(T) << shift
    (chunk & mask) >> shift |> Bool
end

function nchunks(::Type{T}, l) where T
    ((l - 1) >> nshifts(T)) + 1
end
nbits(::Type{T}) where T = sizeof(T) * 8
function nshifts(::Type{T}) where T
    n = nbits(T)
    i = 1
    shifts = 0
    while i & n === 0
        i <<= 1
        shifts += 1
    end
    return shifts 
end



function Base.iterate(s::SBitArray{S,C,T}, state=(1, s.chunks[1])) where {S,C,T}
    (ind, val::T) = state
    size = fieldtypes(S)
    L = prod(size)
    if ind > L
        nothing
    else
        return @inbounds (readbit(val, ind),
            (ind+1, mod1(ind, nbits(T)) == 1 ? S.chunks[nchunks(T, ind)] : val))
    end
end

function dot(a::SBitArray, b::AbstractVector{T}) where T
    s = zero(T)
    for (bool, val) in zip(a, b)
        if bool
            s += val
        end
    end
    return s
end

end # module
