"""
    @reverse_from_forward signature

Define a reverse rule for a given `signature` (function type + argument types) from an existing (primitive or derived) forward rule.

# Example

    @reverse_from_forward Tuple{typeof(f), Float64}

This forces function calls `f(::Float64)` to be differentiated in forward mode inside any reverse-mode procedure.

!!! warning
    This macro is still experimental and has very strict limitations:
        - The signature is limited to a function with one argument
        - The function must have input and output of type `<:Base.IEEEFloat` or `Array{<:Base.IEEEFloat}`
        - The function itself must not close over any data (its tangent type must be `NoTangent`)
"""
macro reverse_from_forward(sig)
    if !(Meta.isexpr(sig, :curly) && sig.args[1] == :Tuple)
        throw(
            ArgumentError(
                "The provided signature must be of the form `Tuple{typeof(f), ...}`."
            ),
        )
    end
    # first two arguments are :Tuple and :(typeof(f))
    nargs = length(sig.args) - 2
    return quote
        @is_primitive DefaultCtx ReverseMode $(esc(sig))
        $(_reverse_from_forward(sig, Val(nargs)))
    end
end

function _reverse_from_forward(sig::Expr, ::Val{N}) where {N}
    throw(
        ArgumentError(
            "`Mooncake.@reverse_from_forward` does not yet support functions with $N arguments.",
        ),
    )
end

function _reverse_from_forward(sig::Expr, ::Val{1})
    F = sig.args[2]
    X = sig.args[3]
    return quote
        function Mooncake.rrule!!(
            f_codual::CoDual{<:$(esc(F))}, x_codual::CoDual{<:$(esc(X))}
        )
            if tangent_type($(esc(F))) != NoTangent
                throw(
                    ArgumentError(
                        "`Mooncake.@reverse_from_forward` does not support functions which close over data.",
                    ),
                )
            end
            return _forward_mode_rrule!!(f_codual, x_codual)
        end
    end
end

function _forward_mode_rrule!!(f_codual::CoDual, x_codual::CoDual{X}) where {X}
    throw(
        ArgumentError(
            "`Mooncake.@reverse_from_forward` does not support functions with input type `$X`.",
        ),
    )
end

function _forward_mode_rrule!!(f_codual::CoDual, x_codual::CoDual{<:IEEEFloat})
    f = primal(f_codual)
    x = primal(x_codual)

    f_dual = Dual(f, NoTangent())
    x_dual = Dual(x, one(x))
    _frule!! = build_frule(f_dual, x_dual)
    y_dual = _frule!!(f_dual, x_dual)

    y = primal(y_dual)
    der = tangent(y_dual)

    df_rdata = NoRData()
    dy_fdata = fdata(zero_tangent(y))

    function forward_mode_scalar_pullback(dy_rdata::IEEEFloat)
        dx_rdata = der * dy_rdata
        return df_rdata, dx_rdata
    end

    function forward_mode_array_pullback(::NoRData)
        dx_rdata = dot(der, dy_fdata)
        return df_rdata, dx_rdata
    end

    if y isa IEEEFloat
        return CoDual(y, dy_fdata), forward_mode_scalar_pullback
    elseif y isa Array{<:IEEEFloat}
        return CoDual(y, dy_fdata), forward_mode_array_pullback
    else
        throw(
            ArgumentError(
                "`Mooncake.@reverse_from_forward` does not support output type `$(typeof(y))` with input type `$(typeof(x))`.",
            ),
        )
    end
end

function _forward_mode_rrule!!(f_codual::CoDual, x_codual::CoDual{<:Array{<:IEEEFloat}})
    f = primal(f_codual)
    x = primal(x_codual)
    dx_fdata = tangent(x_codual)

    f_dual = Dual(f, NoTangent())
    df_rdata = NoRData()

    _frule!! = build_frule(f_dual, Dual(x, zero(x)))
    y = primal(_frule!!(f_dual, Dual(x, zero(x))))  # TODO: one call too many
    dy_fdata = fdata(zero_tangent(y))

    function forward_mode_scalar_pullback(dy_rdata::IEEEFloat)
        dx_fdata .+= map(eachindex(x)) do i
            x_dual = Dual(x, basis(x, i))
            y_dual = _frule!!(f_dual, x_dual)
            der = tangent(y_dual)
            return der * dy_rdata
        end
        dx_rdata = NoRData()
        return df_rdata, dx_rdata
    end

    function forward_mode_array_pullback(_dy_rdata::NoRData)
        dx_fdata .+= map(eachindex(x)) do i
            x_dual = Dual(x, basis(x, i))
            y_dual = _frule!!(f_dual, x_dual)
            der = tangent(y_dual)
            return dot(der, dy_fdata)
        end
        dx_rdata = NoRData()
        return df_rdata, dx_rdata
    end

    if y isa IEEEFloat
        return CoDual(y, dy_fdata), forward_mode_scalar_pullback
    elseif y isa Array{<:IEEEFloat}
        return CoDual(y, dy_fdata), forward_mode_array_pullback
    else
        throw(
            ArgumentError(
                "`Mooncake.@reverse_from_forward` does not support output type `$(typeof(y))` with input type `$(typeof(x))`.",
            ),
        )
    end
end

function basis(a::Array{<:IEEEFloat}, i)
    # TODO: fix for immutable arrays
    # TODO: fix for GPU arrays
    b = zeros(eltype(a), size(a))
    b[i] = oneunit(eltype(b))
    return b
end
