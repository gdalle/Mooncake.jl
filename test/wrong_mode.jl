using LinearAlgebra
using Mooncake
using Mooncake:
    Dual,
    DefaultCtx,
    ForwardMode,
    ReverseMode,
    @is_primitive,
    @reverse_from_forward,
    prepare_pullback_cache,
    value_and_pullback!!,
    primal,
    tangent
using Mooncake.TestUtils: test_rule
using StableRNGs
using Test

## Working

# we define rules for fi that pretend fi_true was run instead, to check that Mooncake doesn't recurse inside the function
# we still use fi_true in the Mooncake.TestUtils.test_rule

cube(x) = x^3
fourthpow(x) = x^4

f1(x::Float64) = sin(x)
f1_true(x::Float64) = cube(x)

f2(x::Float64) = [sin(x), cos(x)]
f2_true(x::Float64) = [cube(x), fourthpow(x)]

f3(x::Vector{Float64}) = sum(sin, x)
f3_true(x::Vector{Float64}) = sum(cube, x)

f4(x::Vector{Float64}) = map(sin, x)
f4_true(x::Vector{Float64}) = map(cube, x)

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f1_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2),Float64}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f2_true),Float64}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f3_true),Vector{Float64}}

@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4),Vector{Float64}}
@is_primitive DefaultCtx ForwardMode Tuple{typeof(f4_true),Vector{Float64}}

function Mooncake.frule!!(
    ::Union{Dual{typeof(f1)},Dual{typeof(f1_true)}}, x_dual::Dual{Float64}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = x^3
    dy = 3x^2 * dx
    return Dual(y, dy)
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f2)},Dual{typeof(f2_true)}}, x_dual::Dual{Float64}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y, z = x^3, x^4
    dy, dz = 3x^2 * dx, 4x^3 * dx
    return Dual([y, z], [dy, dz])
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f3)},Dual{typeof(f3_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = sum(_x -> _x^3, x)
    dy = dot(map(_x -> 3 * _x^2, x), dx)
    return Dual(y, dy)
end

function Mooncake.frule!!(
    ::Union{Dual{typeof(f4)},Dual{typeof(f4_true)}}, x_dual::Dual{Vector{Float64}}
)
    x, dx = primal(x_dual), tangent(x_dual)
    y = map(_x -> _x^3, x)
    J = diagm(map(_x -> 3 * _x^2, x))
    dy = J * dx
    return Dual(y, dy)
end

@reverse_from_forward Tuple{typeof(f1),Float64}
@reverse_from_forward Tuple{typeof(f1_true),Float64}

@reverse_from_forward Tuple{typeof(f2),Float64}
@reverse_from_forward Tuple{typeof(f2_true),Float64}

@reverse_from_forward Tuple{typeof(f3),Vector{Float64}}
@reverse_from_forward Tuple{typeof(f3_true),Vector{Float64}}

@reverse_from_forward Tuple{typeof(f4),Vector{Float64}}
@reverse_from_forward Tuple{typeof(f4_true),Vector{Float64}}

@testset verbose = true "Working cases" begin
    @testset "Scalar to scalar" begin
        x = 5.0
        dy = 7.0
        cache = prepare_pullback_cache(f1, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f1, x)
        @test val == x^3
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy * 3x^2
        test_rule(StableRNG(63), f1_true, x)
    end
    @testset "Scalar to array" begin
        x = 5.0
        dy = [7.0, 11.0]
        cache = prepare_pullback_cache(f2, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f2, x)
        @test val == [x^3, x^4]
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy[1] * 3x^2 + dy[2] * 4x^3
        test_rule(StableRNG(63), f2_true, x)
    end
    @testset "Array to scalar" begin
        x = [5.0, 13.0]
        dy = 7.0
        cache = prepare_pullback_cache(f3, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f3, x)
        @test val == sum(cube, x)
        @test pb[1] == Mooncake.NoTangent()
        @test pb[2] == dy .* map(_x -> 3 * _x^2, x)
        test_rule(StableRNG(63), f3_true, x)
    end
    @testset "Array to array" begin
        x = [5.0, 13.0]
        dy = [7.0, 11.0]
        cache = prepare_pullback_cache(f4, zero(x))
        val, pb = value_and_pullback!!(cache, dy, f4, x)
        @test val == map(cube, x)
        @test pb[1] == Mooncake.NoTangent()
        test_rule(StableRNG(63), f4_true, x)
    end
end;

## Failing

# unsupported types
f5(x::Float64) = (x, 2x)
f6(x::Vector{Float64}) = (x[1], 2x[1])
f7(x::Tuple{Float64}) = x[1]

@reverse_from_forward(Tuple{typeof(f5),Float64})
@reverse_from_forward(Tuple{typeof(f6),Vector{Float64}})
@reverse_from_forward(Tuple{typeof(f7),Tuple{Float64}})

struct Multiplier
    a::Float64
end
(m::Multiplier)(x) = m.a * x

@reverse_from_forward Tuple{Multiplier,Float64}

@testset verbose = true "Failing cases" begin
    @testset "Wrong expression (not a Tuple{...})" begin
        @test_throws "LoadError: ArgumentError: The provided signature must be of the form `Tuple{typeof(f), ...}`." @eval @reverse_from_forward(
            :(f3(x::Float64))
        )
    end
    @testset "Wrong number of arguments" begin
        @test_throws "LoadError: ArgumentError: `Mooncake.@reverse_from_forward` does not yet support functions with 2 arguments." @eval @reverse_from_forward(
            Tuple{typeof(f4),Float64,Float64}
        )
    end
    @testset "Closures" begin
        @test_throws "ArgumentError: `Mooncake.@reverse_from_forward` does not support functions which close over data." prepare_gradient_cache(
            Multiplier(2.0), 5.0
        )
    end
    @testset "Unsupported output types" begin
        @test_throws "ArgumentError: `Mooncake.@reverse_from_forward` does not support output type `Tuple{Float64, Float64}` with input type `Float64`." prepare_pullback_cache(
            f5, 5.0
        )
        @test_throws "ArgumentError: `Mooncake.@reverse_from_forward` does not support output type `Tuple{Float64, Float64}` with input type `Vector{Float64}`." prepare_pullback_cache(
            f6, [5.0, 13.0]
        )
    end
    @testset "Unsupported input types" begin
        @test_throws "ArgumentError: `Mooncake.@reverse_from_forward` does not support functions with input type `Tuple{Float64}`." prepare_pullback_cache(
            f7, (5.0,)
        )
    end
end;
