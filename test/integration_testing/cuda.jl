using CUDA

@testset "cuda" begin

    # Check we can operate on CuArrays.
    test_tangent(
        Xoshiro(123456),
        CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}(undef, 8, 8);
        interface_only=false,
    )

    # Check we can instantiate a CuArray.
    test_rule(
        sr(123456), CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, undef, 256;
        interface_only=true, is_primitive=true,
    )
end
