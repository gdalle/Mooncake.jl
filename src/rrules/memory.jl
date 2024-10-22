# This file was introduced as part of the transition from 1.10 to 1.11. Its purpose is to
# ensure that Mooncake can handle the new implementation of `Array`s. This implementation
# relies on the new `Memory` and `MemoryRef` types (aliases for specific parametrisations of
# `GenericMemory` and `GenericMemoryRef`). Consequently, the code here will make little
# sense unless you are familiar with these types, and how they relate to `Array`s.
# Fortunately, Oscar Smith and Jameson Nash gave an excellent talk at JuliaCon 2024 on
# exactly this topic, which you can find here: https://www.youtube.com/watch?v=L6BFQ1d8xNs .

#
# Memory
#

# Tangent Interface Implementation

const Maybe{T} = Union{Nothing, T}

tangent_type(::Type{<:Memory{P}}) where {P} = Memory{tangent_type(P)}

function zero_tangent_internal(x::Memory{P}, stackdict::Maybe{IdDict}) where {P}
    T = tangent_type(typeof(x))
    haskey(stackdict, x) && return stackdict[x]::T

    t = T(undef, length(x))
    stackdict[x] = t
    return _map_if_assigned!(Base.Fix2(zero_tangent_internal, stackdict), t, x)::T
end

function randn_tangent_internal(rng::AbstractRNG, x::Memory, stackdict::Maybe{IdDict})
    T = tangent_type(typeof(x))
    haskey(stackdict, x) && return stackdict[x]::T

    t = T(undef, length(x))
    stackdict[t] = t
    return _map_if_assigned!(x -> randn_tangent_internal(rng, x, stackdict), t, x)::T
end

function TestUtils.has_equal_data_internal(
    x::Memory{P}, y::Memory{P}, equal_undefs::Bool, d::Dict{Tuple{UInt, UInt}, Bool}
) where {P}
    length(x) == length(y) || return false
    id_pair = (objectid(x), objectid(y))
    haskey(d, id_pair) && return d[id_pair]

    d[id_pair] = true
    equality = map(1:length(x)) do n
        if isassigned(x, n) != isassigned(y, n)
            return !equal_undefs
        elseif !isassigned(x, n)
            return true
        else
            return TestUtils.has_equal_data_internal(x[n], y[n], equal_undefs, d)
        end
    end
    return all(equality)
end

function increment!!(x::Memory{P}, y::Memory{P}) where {P}
    return x === y ? x : _map_if_assigned!(increment!!, x, x, y)
end

set_to_zero!!(x::Memory) = _map_if_assigned!(set_to_zero!!, x, x)

function _add_to_primal(p::Memory{P}, t::Memory) where {P}
    return _map_if_assigned!(_add_to_primal, Memory{P}(undef, length(p)), p, t)
end

function _diff(p::Memory{P}, q::Memory{P}) where {P}
    return _map_if_assigned!(_diff, Memory{tangent_type(P)}(undef, length(p)), p ,q)
end

function _dot(t::Memory{T}, s::Memory{T}) where {T}
    isbitstype(T) && return sum(_map(_dot, t, s))
    return sum(
        _map(eachindex(t)) do n
            (isassigned(t, n) && isassigned(s, n)) ? _dot(t[n], s[n]) : 0.0
        end
    )
end

function _scale(a::Float64, t::Memory{T}) where {T}
    return _map_if_assigned!(Base.Fix1(_scale, a), Memory{T}(undef, length(t)), t)
end

import .TestUtils: populate_address_map!
function populate_address_map!(m::TestUtils.AddressMap, p::Memory, t::Memory)
    k = pointer_from_objref(p)
    v = pointer_from_objref(t)
    haskey(m, k) && (@assert m[k] == v)
    m[k] = v
    foreach(n -> isassigned(p, n) && populate_address_map!(m, p[n], t[n]), eachindex(p))
    return m
end

# FData / RData Interface Implementation

tangent_type(::Type{F}, ::Type{NoRData}) where {F<:Memory} = F

tangent(f::Memory, ::NoRData) = f

function _verify_fdata_value(p::Memory{P}, f::Memory{T}) where {P, T}
    @assert length(p) == length(f)
    return nothing
end

# Rules

@is_primitive(
    MinimalCtx, Tuple{typeof(unsafe_copyto!), MemoryRef{P}, MemoryRef{P}, Int} where {P}
)
function rrule!!(
    ::CoDual{typeof(unsafe_copyto!)},
    dest::CoDual{MemoryRef{P}},
    src::CoDual{MemoryRef{P}},
    _n::CoDual{Int},
) where {P}
    n = primal(_n)

    # Copy state of primal and fdata of dest.
    dest_primal_copy = memoryref(Memory{P}(undef, n))
    dest_fdata_copy = memoryref(Memory{tangent_type(P)}(undef, n))
    unsafe_copyto!(dest_primal_copy, dest.x, n)
    unsafe_copyto!(dest_fdata_copy, dest.dx, n)

    # Apply primal computation to both primal and fdata.
    unsafe_copyto!(dest.x, src.x, n)
    unsafe_copyto!(dest.dx, src.dx, n)

    function unsafe_copyto!_adjoint(::NoRData)

        # Increment tangents in src by values in dest.
        tmp = Memory{eltype(dest.dx)}(undef, n)
        unsafe_copyto!(memoryref(tmp), dest.dx, n)

        # Restore state of `dest`.
        unsafe_copyto!(dest.x, dest_primal_copy, n)
        unsafe_copyto!(dest.dx, dest_fdata_copy, n)

        # Increment gradients.
        @inbounds for i in 1:n
            src_ref = memoryref(src.dx, i)
            src_ref[] = increment!!(src_ref[], memoryref(tmp, i)[])
        end

        return ntuple(_ -> NoRData(), 4)
    end
    return dest, unsafe_copyto!_adjoint
end

#
# MemoryRef
#

# Tangent Interface Implementation

tangent_type(::Type{<:MemoryRef{P}}) where {P} = MemoryRef{tangent_type(P)}

function zero_tangent_internal(x::MemoryRef, stackdict::Maybe{IdDict})
    t_mem = zero_tangent_internal(x.mem, stackdict)::Memory
    return memoryref(t_mem, Core.memoryrefoffset(x))
end

function randn_tangent_internal(rng::AbstractRNG, x::MemoryRef, stackdict::Maybe{IdDict})
    t_mem = randn_tangent_internal(rng, x.mem, stackdict)::Memory
    return memoryref(t_mem, Core.memoryrefoffset(x))
end

function TestUtils.has_equal_data_internal(
    x::MemoryRef{P}, y::MemoryRef{P}, equal_undefs::Bool, d::Dict{Tuple{UInt, UInt}, Bool}
) where {P}
    equal_refs = Core.memoryrefoffset(x) == Core.memoryrefoffset(y)
    equal_data = TestUtils.has_equal_data_internal(x.mem, y.mem, equal_undefs, d)
    return equal_refs && equal_data
end

function increment!!(x::MemoryRef{P}, y::MemoryRef{P}) where {P}
    return memoryref(increment!!(x.mem, y.mem), Core.memoryrefoffset(x))
end

function set_to_zero!!(x::MemoryRef)
    set_to_zero!!(x.mem)
    return x
end

function _add_to_primal(p::MemoryRef, t::MemoryRef)
    return memoryref(_add_to_primal(p.mem, t.mem), Core.memoryrefoffset(p))
end

function _diff(p::MemoryRef{P}, q::MemoryRef{P}) where {P}
    @assert Core.memoryrefoffset(p) == Core.memoryrefoffset(q)
    return memoryref(_diff(p.mem, q.mem), Core.memoryrefoffset(p))
end

function _dot(t::MemoryRef{T}, s::MemoryRef{T}) where {T}
    @assert Core.memoryrefoffset(t) == Core.memoryrefoffset(s)
    return _dot(t.mem, s.mem)
end

_scale(a::Float64, t::MemoryRef) = memoryref(_scale(a, t.mem), Core.memoryrefoffset(t))

function populate_address_map!(m::TestUtils.AddressMap, p::MemoryRef, t::MemoryRef)
    return populate_address_map!(m, p.mem, t.mem)
end

# FData / RData Interface Implementation

fdata_type(::Type{<:MemoryRef{T}}) where {T} = MemoryRef{T}

rdata_type(::Type{<:MemoryRef}) = NoRData

tangent_type(::Type{<:MemoryRef{T}}, ::Type{NoRData}) where {T} = MemoryRef{T}

tangent(f::MemoryRef, ::NoRData) = f

function _verify_fdata_value(p::MemoryRef{P}, f::MemoryRef{T}) where {P, T}
    _verify_fdata_value(p.mem, f.mem)
end

#
# Rules for `Memory` and `MemoryRef`s
#

_val(::Val{c}) where {c} = c

using Core: memoryref_isassigned, memoryrefget, memoryrefset!, memoryrefnew, memoryrefoffset

@zero_adjoint(
    MinimalCtx, Tuple{typeof(memoryref_isassigned), GenericMemoryRef, Symbol, Bool}
)

@inline function lmemoryrefget(
    x::MemoryRef, ::Val{ordering}, ::Val{boundscheck}
) where {ordering, boundscheck}
    return memoryrefget(x, ordering, boundscheck)
end

@is_primitive MinimalCtx Tuple{typeof(lmemoryrefget), MemoryRef, Val, Val}
@inline function rrule!!(
    ::CoDual{typeof(lmemoryrefget)},
    x::CoDual{<:MemoryRef},
    _ordering::CoDual{<:Val},
    _boundscheck::CoDual{<:Val},
)
    ordering = primal(_ordering)
    bc = primal(_boundscheck)
    dx = x.dx
    function lmemoryrefget_adjoint(dy)
        new_tangent = increment_rdata!!(memoryrefget(dx, _val(ordering), _val(bc)), dy)
        memoryrefset!(dx, new_tangent, _val(ordering), _val(bc))
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    y = memoryrefget(x.x, _val(ordering), _val(bc))
    dy = fdata(memoryrefget(x.dx, _val(ordering), _val(bc)))
    return CoDual(y, dy), lmemoryrefget_adjoint
end

@inline Base.@propagate_inbounds function rrule!!(
    ::CoDual{typeof(memoryrefget)},
    x::CoDual{<:MemoryRef},
    _ordering::CoDual{Symbol},
    _boundscheck::CoDual{Bool},
)
    out, adj = rrule!!(
        zero_fcodual(lmemoryrefget),
        x,
        zero_fcodual(Val(primal(_ordering))),
        zero_fcodual(Val(primal(_boundscheck))),
    )
    memoryrefget_adjoint(dy) = adj(dy)
    return out, memoryrefget_adjoint
end

# Core.memoryrefmodify!

@inline function rrule!!(f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:Memory})
    return CoDual(memoryrefnew(x.x), memoryrefnew(x.dx)), NoPullback(f, x)
end

@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)}, x::CoDual{<:MemoryRef}, ii::CoDual{Int}
)
    return CoDual(memoryrefnew(x.x, ii.x), memoryrefnew(x.dx, ii.x)), NoPullback(f, x, ii)
end

@inline function rrule!!(
    f::CoDual{typeof(memoryrefnew)},
    x::CoDual{<:MemoryRef},
    ii::CoDual{Int},
    boundscheck::CoDual{Bool},
)
    y = memoryrefnew(x.x, ii.x, boundscheck.x)
    dy = memoryrefnew(x.dx, ii.x, boundscheck.x)
    return CoDual(y, dy), NoPullback(f, x, ii, boundscheck)
end

@zero_adjoint MinimalCtx Tuple{typeof(memoryrefoffset), GenericMemoryRef}

# Core.memoryrefreplace!

@inline function lmemoryrefset!(
    x::MemoryRef, value, ::Val{ordering}, ::Val{boundscheck}
) where {ordering, boundscheck}
    return memoryrefset!(x, value, ordering, boundscheck)
end

@is_primitive MinimalCtx Tuple{typeof(lmemoryrefset!), MemoryRef, Any, Val, Val}

@inline function rrule!!(
    ::CoDual{typeof(lmemoryrefset!)},
    x::CoDual{<:MemoryRef{P}, <:MemoryRef{V}},
    value::CoDual,
    _ordering::CoDual{<:Val},
    _boundscheck::CoDual{<:Val},
) where {P, V}
    ordering = primal(_ordering)
    bc = primal(_boundscheck)

    isbitstype(P) && return isbits_lmemoryrefset!_rule(x, value, ordering, bc)

    to_save = isassigned(x.x)
    old_x = Ref{Tuple{P, V}}()
    if to_save
        old_x[] = (
            memoryrefget(x.x, _val(ordering), _val(bc)),
            memoryrefget(x.dx, _val(ordering), _val(bc)),
        )
    end

    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    dx = x.dx
    memoryrefset!(dx, tangent(value.dx, zero_rdata(value.x)), _val(ordering), _val(bc))
    function lmemoryrefset_adjoint(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(dx, _val(ordering), _val(bc))))
        if to_save
            memoryrefset!(x.x, old_x[][1], _val(ordering), _val(bc))
            memoryrefset!(dx, old_x[][2], _val(ordering), _val(bc))
        end
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, lmemoryrefset_adjoint
end

function isbits_lmemoryrefset!_rule(x::CoDual, value::CoDual, ordering::Val, bc::Val)
    old_x = (
        memoryrefget(x.x, _val(ordering), _val(bc)),
        memoryrefget(x.dx, _val(ordering), _val(bc)),
    )
    memoryrefset!(x.x, value.x, _val(ordering), _val(bc))
    memoryrefset!(x.dx, zero_tangent(value.x), _val(ordering), _val(bc))

    function isbits_lmemoryrefset!_adjoint(dy)
        dvalue = increment!!(dy, rdata(memoryrefget(x.dx, _val(ordering), _val(bc))))
        memoryrefset!(x.x, old_x[1], _val(ordering), _val(bc))
        memoryrefset!(x.dx, old_x[2], _val(ordering), _val(bc))
        return NoRData(), NoRData(), dvalue, NoRData(), NoRData()
    end
    return value, isbits_lmemoryrefset!_adjoint
end

@inline function rrule!!(
    ::CoDual{typeof(memoryrefset!)},
    x::CoDual{<:MemoryRef{P}, <:MemoryRef{V}},
    value::CoDual,
    ordering::CoDual{Symbol},
    boundscheck::CoDual{Bool},
) where {P, V}
    y, adj = rrule!!(
        zero_fcodual(lmemoryrefset!),
        x,
        value,
        zero_fcodual(Val(primal(ordering))),
        zero_fcodual(Val(primal(boundscheck))),
    )
    memoryrefset_adjoint(dy) = adj(dy)
    return y, memoryrefset_adjoint
end

# Core.memoryrefsetonce!
# Core.memoryrefswap!
# Core.set_binding_type!

# _new_ and _new_-adjacent rules for Memory, MemoryRef, and Array.

@zero_adjoint MinimalCtx Tuple{Type{<:Memory}, UndefInitializer, Int}

function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{MemoryRef{P}}},
    ptr_or_offset::CoDual{Ptr{Nothing}},
    mem::CoDual{Memory{P}},
) where {P}
    y = _new_(MemoryRef{P}, ptr_or_offset.x, mem.x)
    dy = _new_(MemoryRef{tangent_type(P)}, bitcast(Ptr{Nothing}, ptr_or_offset.dx), mem.dx)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function rrule!!(
    ::CoDual{typeof(_new_)},
    ::CoDual{Type{Array{P, N}}},
    ref::CoDual{MemoryRef{P}},
    size::CoDual{<:NTuple{N, Int}},
) where {P, N}
    y = _new_(Array{P, N}, ref.x, size.x)
    dy = _new_(Array{tangent_type(P), N}, ref.dx, size.x)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

# getfield / lgetfield rules for Memory, MemoryRef, and Array.

function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:Memory},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name, order}
    y = getfield(primal(x), name, order)
    wants_length = name === 1 || name === :length
    dy = wants_length ? NoFData() : bitcast(Ptr{NoTangent}, x.dx.ptr)
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:MemoryRef},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name, order}
    y = getfield(primal(x), name, order)
    wants_offset = name === 1 || name === :ptr_or_offset
    dy = wants_offset ? bitcast(Ptr{NoTangent}, x.dx.ptr_or_offset) : x.dx.mem
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

function rrule!!(
    ::CoDual{typeof(lgetfield)},
    x::CoDual{<:Array},
    ::CoDual{Val{name}},
    ::CoDual{Val{order}},
) where {name, order}
    y = getfield(primal(x), name, order)
    wants_size = name === 2 || name === :size
    dy = wants_size ? NoFData() : x.dx.ref
    return CoDual(y, dy), NoPullback(ntuple(_ -> NoRData(), 4))
end

const _MemTypes = Union{Memory, MemoryRef, Array}

function rrule!!(f::CoDual{typeof(lgetfield)}, x::CoDual{<:_MemTypes}, name::CoDual{<:Val})
    y, adj = rrule!!(f, x, name, zero_fcodual(Val(:not_atomic)))
    ternary_lgetfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_lgetfield_adjoint
end

function rrule!!(
    ::CoDual{typeof(getfield)}, x::CoDual{<:_MemTypes},
    name::CoDual{<:Union{Int, Symbol}},
    order::CoDual{Symbol},
)
    y, adj = rrule!!(
        zero_fcodual(lgetfield),
        x,
        zero_fcodual(Val(primal(name))),
        zero_fcodual(Val(primal(order))),
    )
    getfield_adjoint(dy) = adj(dy)
    return y, getfield_adjoint
end

function rrule!!(
    f::CoDual{typeof(getfield)}, x::CoDual{<:_MemTypes}, name::CoDual{<:Union{Int, Symbol}}
)
    y, adj = rrule!!(f, x, name, zero_fcodual(:not_atomic))
    ternary_getfield_adjoint(dy) = adj(dy)[1:3]
    return y, ternary_getfield_adjoint
end

@inline function rrule!!(
    ::CoDual{typeof(lsetfield!)}, value::CoDual{<:Array}, ::CoDual{Val{name}}, x::CoDual
) where {name}
    old_x = getfield(value.x, name)
    old_dx = getfield(value.dx, name)
    setfield!(value.x, name, x.x)
    setfield!(value.dx, name, (name === :size || name === 2) ? x.x : x.dx)
    function array_lsetfield!_adjoint(::NoRData)
        setfield!(value.x, name, old_x)
        setfield!(value.dx, name, old_dx)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, array_lsetfield!_adjoint
end

# Misc. other rules which are required for correctness.

@is_primitive MinimalCtx Tuple{typeof(copy), Array}
function rrule!!(::CoDual{typeof(copy)}, a::CoDual{<:Array})
    dx = tangent(a)
    dy = copy(dx)
    y = CoDual(copy(primal(a)), dy)
    function copy_pullback!!(::NoRData)
        increment!!(dx, dy)
        return NoRData(), NoRData()
    end
    return y, copy_pullback!!
end

@is_primitive MinimalCtx Tuple{typeof(fill!), Array{<:Union{UInt8, Int8}}, Integer}
@is_primitive MinimalCtx Tuple{typeof(fill!), Memory{<:Union{UInt8, Int8}}, Integer}
function rrule!!(
    ::CoDual{typeof(fill!)}, a::CoDual{T}, x::CoDual{<:Integer}
) where {V<:Union{UInt8, Int8}, T<:Union{Array{V}, Memory{V}}}
    pa = primal(a)
    old_value = copy(pa)
    fill!(pa, primal(x))
    function fill!_pullback!!(::NoRData)
        pa .= old_value
        return NoRData(), NoRData(), NoRData()
    end
    return a, fill!_pullback!!
end

# Test cases

function _mems()

    # Set up memory with an undefined element.
    mem_with_single_undef = Memory{Memory{Int}}(undef, 2)
    mem_with_single_undef[1] = fill!(Memory{Int}(undef, 4), 2)

    # Return a collection of test cases.
    mems = [
        (fill!(Memory{Float64}(undef, 10), 0.0)),
        (fill!(Memory{Int}(undef, 5), 1)),
        (Memory{Vector{Float64}}([randn(1), randn(3)])),
        (Memory{Vector{Float64}}(undef, 3)),
        (Memory{Any}(randn(3))),
        (mem_with_single_undef),
    ]
    sample_values = [1.0, 3, randn(2), randn(2), 5.0, Memory{Int}(undef, 5)]
    return mems, sample_values
end

function _mem_refs()
    mems_1, sample_values_1 = _mems()
    mems_2, sample_values_2 = _mems()
    mem_refs = vcat([memoryref(m) for m in mems_1], [memoryref(m, 2) for m in mems_2])
    return mem_refs, vcat(sample_values_1, sample_values_2)
end

function generate_data_test_cases(rng_ctor, ::Val{:memory})
    arrays = [
        randn(2),
    ]
    return vcat(_mems()[1], _mem_refs()[1], arrays)
end

function generate_hand_written_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    mems, _ = _mems()
    mem_refs, sample_mem_ref_values = _mem_refs()
    test_cases = vcat(

        # Rules for `Memory`
        (true, :stability, nothing, Memory{Float64}, undef, 5),
        (true, :stability, nothing, Memory{Memory{Float64}}, undef, 5),
        [(false, :stability_and_allocs, nothing, lgetfield, m, Val(:length)) for m in mems],
        [(false, :stability_and_allocs, nothing, lgetfield, m, Val(1)) for m in mems],
        [(false, :none, nothing, getfield, m, :length) for m in mems],
        [(false, :none, nothing, getfield, m, 1) for m in mems],

        # Rules for `MemoryRef`
        [(false, :none, nothing, memoryref_isassigned, mem_ref, :not_atomic, bc) for
            mem_ref in mem_refs for bc in [false, true]
        ],
        [(false, :none, nothing, memoryrefget, mem_ref, :not_atomic, bc) for
            mem_ref in filter(isassigned, mem_refs) for bc in [false, true]
        ],
        [(false, :none, nothing, memoryrefnew, mem) for mem in mems],
        [(false, :none, nothing, memoryrefnew, mem, 1) for mem in mem_refs],
        [(false, :none, nothing, memoryrefnew, mem, 1, bc) for
            mem in mem_refs for bc in [false, true]
        ],
        [(false, :none, nothing, memoryrefoffset, mem_ref) for mem_ref in mem_refs],
        [
            (false, :none, nothing, lmemoryrefset!, mem_ref, sample_value, Val(:not_atomic), bc) for
            (mem_ref, sample_value) in zip(mem_refs, sample_mem_ref_values) for
            bc in [Val(false), Val(true)]
        ],
        [
            (false, :none, nothing, memoryrefset!, mem_ref, sample_value, :not_atomic, bc) for
            (mem_ref, sample_value) in zip(mem_refs, sample_mem_ref_values) for
            bc in [false, true]
        ],
        (false, :stability, nothing, unsafe_copyto!, randn(10).ref, randn(8).ref, 5),
        (
            false, :stability, nothing,
            unsafe_copyto!,
            memoryref(randn(10).ref, 2),
            memoryref(randn(8).ref, 3),
            4,
        ),
        (
            false, :stability, nothing,
            unsafe_copyto!,
            [randn(10), randn(5)].ref,
            [randn(10), randn(3)].ref,
            2,
        ),

        # Rules for `Array`
        (false, :stability, nothing, _new_, Vector{Float64}, randn(10).ref, (10, )),
        (
            false, :stability, nothing,
            _new_,
            Vector{Vector{Float64}},
            [randn(10), randn(5)].ref,
            (2, ),
        ),
        (
            false, :none, nothing,
            _new_,
            Vector{Any},
            [1, randn(5)].ref,
            (2, ),
        ),
        (false, :stability, nothing, _new_, Matrix{Float64}, randn(12).ref, (4, 3)),
        (false, :stability, nothing, _new_, Array{Float64, 3}, randn(12).ref, (4, 1, 3)),
        [
            (false, :stability, nothing, lgetfield, randn(10), f) for
                f in [Val(:ref), Val(:size), Val(1), Val(2)]
        ],
        [
            (false, :none, nothing, getfield, randn(10), f) for
                f in [:ref, :size, 1, 2]
        ],
        (false, :stability_and_allocs, nothing, lsetfield!, randn(10), Val(:ref), randn(10).ref),
        (false, :stability_and_allocs, nothing, lsetfield!, randn(10), Val(1), randn(10).ref),
        (false, :stability_and_allocs, nothing, lsetfield!, randn(10), Val(:size), (10, )),
        (false, :stability_and_allocs, nothing, lsetfield!, randn(10), Val(2), (10, )),
        (false, :none, nothing, setfield!, randn(10), :ref, randn(10).ref),
        (false, :none, nothing, setfield!, randn(10), 1, randn(10).ref),
        (false, :none, nothing, setfield!, randn(10), :size, (10, )),
        (false, :none, nothing, setfield!, randn(10), 2, (10, )),
    )
    memory = Any[]
    return test_cases, memory
end

function generate_derived_rrule!!_test_cases(rng_ctor, ::Val{:memory})
    x = Memory{Float64}(randn(10))
    test_cases = Any[
        (true, :none, nothing, Array{Float64, 0}, undef),
        (true, :none, nothing, Array{Float64, 1}, undef, 5),
        (true, :none, nothing, Array{Float64, 2}, undef, 5, 4),
        (true, :none, nothing, Array{Float64, 3}, undef, 5, 4, 3),
        (true, :none, nothing, Array{Float64, 4}, undef, 5, 4, 3, 2),
        (true, :none, nothing, Array{Float64, 5}, undef, 5, 4, 3, 2, 1),
        (true, :none, nothing, Array{Float64, 0}, undef, ()),
        (true, :none, nothing, Array{Float64, 4}, undef, (2, 3, 4, 5)),
        (true, :none, nothing, Array{Float64, 5}, undef, (2, 3, 4, 5, 6)),
        (false, :none, nothing, copy, randn(5, 4)),
        (false, :none, nothing, Base._deletebeg!, randn(5), 0),
        (false, :none, nothing, Base._deletebeg!, randn(5), 2),
        (false, :none, nothing, Base._deletebeg!, randn(5), 5),
        (false, :none, nothing, Base._deleteend!, randn(5), 2),
        (false, :none, nothing, Base._deleteend!, randn(5), 5),
        (false, :none, nothing, Base._deleteend!, randn(5), 0),
        (false, :none, nothing, Base._deleteat!, randn(5), 2, 2),
        (false, :none, nothing, Base._deleteat!, randn(5), 1, 5),
        (false, :none, nothing, Base._deleteat!, randn(5), 5, 1),
        (false, :none, nothing, fill!, rand(Int8, 5), Int8(2)),
        (false, :none, nothing, fill!, rand(UInt8, 5), UInt8(2)),
        (false, :none, nothing, fill!, Memory{Int8}(rand(Int8, 5)), Int8(3)),
        (false, :none, nothing, fill!, Memory{UInt8}(rand(UInt8, 5)), UInt8(5)),
        (true, :none, nothing, Base._growbeg!, randn(5), 3),
        (true, :none, nothing, Base._growend!, randn(5), 3),
        (true, :none, nothing, Base._growat!, randn(5), 2, 2),
        (false, :none, nothing, sizehint!, randn(5), 10),
        (false, :none, nothing, unsafe_copyto!, randn(4), 2, randn(3), 1, 2),
        (
            false, :none, nothing,
            unsafe_copyto!, [rand(3) for _ in 1:5], 2, [rand(4) for _ in 1:4], 1, 3,
        ),
        (
            false, :none, nothing,
            unsafe_copyto!, Vector{Any}(undef, 5), 2, Any[rand() for _ in 1:4], 1, 3,
        ),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x, 1), memoryref(x), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x, 2), 3), x),
        (false, :none, nothing, x -> unsafe_copyto!(memoryref(x), memoryref(x, 4), 3), x),
    ]
    memory = Any[]
    return test_cases, memory
end