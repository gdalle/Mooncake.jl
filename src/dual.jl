struct Dual{P,T}
    primal::P
    tangent::T
end

primal(x::Dual) = x.primal
tangent(x::Dual) = x.tangent
Base.copy(x::Dual) = Dual(copy(primal(x)), copy(tangent(x)))
_copy(x::P) where {P<:Dual} = x

zero_dual(x) = Dual(x, zero_tangent(x))

function dual_type(::Type{P}) where {P}
    P == DataType && return Dual
    P isa Union && return Union{dual_type(P.a),dual_type(P.b)}
    P <: UnionAll && return Dual # P is abstract, so we don't know its tangent type.
    return isconcretetype(P) ? Dual{P,tangent_type(P)} : Dual
end

function dual_type(p::Type{Type{P}}) where {P}
    return @isdefined(P) ? Dual{Type{P},NoTangent} : Dual{_typeof(p),NoTangent}
end

_primal(x) = x
_primal(x::Dual) = primal(x)

make_dual(x) = zero_dual(x)
make_dual(x::Dual) = x