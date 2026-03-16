struct ForwardModeRRule!!{FR}
    _frule!!::FR
end

function __verify_sig(rule::ForwardModeRRule!!, fx)
    # TODO: modify `fx`
    return __verify_sig(rule._frule!!, fx)
end

function (forward_mode_rrule!!::ForwardModeRRule!!)(
    f_codual::CoDual{F},
    args_codual::Vararg{<:CoDual{<:Union{IEEEFloat,<:Array{<:IEEEFloat}}},N},  # TODO: relax
) where {F,N}
    (; _frule!!) = forward_mode_rrule!!
    f = primal(f_codual)
    args = map(primal, args_codual)

    if tangent_type(F) != NoTangent
        throw(
            ArgumentError(
                "`Mooncake.@reverse_from_forward` does not support functions which close over data.",
            ),
        )
    end

    f_dual = Dual(f, NoTangent())
    args_dual_zero = map(zero_dual, args)
    y_dual = _frule!!(f_dual, args_dual_zero...)

    y = primal(y_dual)
    y_codual = zero_fcodual(y)

    function forward_mode_pullback(dy_rdata)
        @assert dy_rdata isa rdata_type(typeof(y)) # TODO: check this at compile time
        # compute args rdata
        dargs_rdata = ntuple(Val(N)) do i
            # ntuple ensures unrolling with statically inferrable integers (hopefully)
            if rdata_type(typeof(args[i])) == NoRData
                return NoRData()
            else
                @assert args[i] isa IEEEFloat # TODO: relax
                # create perturbation of scalar argument i
                args_dual_one_i = ntuple(Val(N)) do k
                    k == i ? Dual(args[i], one(args[i])) : args_dual_zero[k]
                end
                # compute partial derivative with respect to argument i
                y_dual_one_i = _frule!!(f_dual, args_dual_one_i...)
                partial_derivative_i = tangent(y_dual_one_i)
                # deduce one component of the pullback using a dot product
                rdata_i =
                    dot(fdata(partial_derivative_i), tangent(y_codual)) +
                    dot(rdata(partial_derivative_i), dy_rdata)
                return rdata_i
            end
        end

        # update args fdata
        ntuple(Val(N)) do i
            # ntuple ensures unrolling with statically inferrable integers (hopefully)
            if tangent(args_codual[i]) isa NoFData
                return nothing
            else
                @assert args[i] isa Array{<:IEEEFloat} # TODO: relax
                # Reuse a single buffer for the basis vector: set b[j]=1, call frule, reset.
                # This avoids one allocation per input dimension.
                b = zero(args[i])
                for j in eachindex(args[i])
                    b[j] = oneunit(eltype(b))
                    args_dual_one_ij = ntuple(Val(N)) do k
                        k == i ? Dual(args[i], b) : args_dual_zero[k]
                    end
                    y_dual_one_ij = _frule!!(f_dual, args_dual_one_ij...)
                    partial_derivative_ij = tangent(y_dual_one_ij)
                    tangent(args_codual[i])[j] +=
                        dot(fdata(partial_derivative_ij), tangent(y_codual)) +
                        dot(rdata(partial_derivative_ij), dy_rdata)
                    b[j] = zero(eltype(b))
                end
                return nothing
            end
        end

        # return function and args rdata
        df_rdata = NoRData()
        return (df_rdata, dargs_rdata...)
    end

    return y_codual, forward_mode_pullback
end


"""
    @reverse_from_forward signature

Define a reverse rule for a given `signature` (function type + argument types) from an existing (primitive or derived) forward rule.

# Example

    @reverse_from_forward Tuple{typeof(f), Float64, Vector{Float64}}

This forces function calls `f(::Float64, ::Vector{Float64})` to be differentiated in forward mode inside any reverse-mode procedure.

!!! warning
    This macro is still experimental and has strict limitations:
        - The function must have all its arguments and its output of type `<:Base.IEEEFloat` or `Array{<:Base.IEEEFloat}`
        - The function must not close over any data (its own tangent type must be `NoTangent`)
        - The function must not mutate any of its arguments. That's because we need to run several forward passes for each argument to mimick a reverse pass, so the state of the arguments must remain unchanged.
"""
macro reverse_from_forward(sig)
    if !(Meta.isexpr(sig, :curly) && sig.args[1] == :Tuple)
        throw(
            ArgumentError(
                "The provided signature must be of the form `Tuple{typeof(f), ...}`."
            ),
        )
    end
    return quote
        @is_primitive DefaultCtx ReverseMode $(esc(sig))
        function Mooncake.build_primitive_rrule(concrete_sig::Type{<:$(esc(sig))})
            interp = get_interpreter(ForwardMode)
            _frule!! = build_frule(interp, concrete_sig)
            return ForwardModeRRule!!(_frule!!)
        end
    end
end
