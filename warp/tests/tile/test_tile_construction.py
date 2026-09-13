# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests tile construction and per-thread tile conversion with 64-thread blocks.

Covers ``tile()``, ``untile()``, ``tile_zeros()``, ``tile_ones()``,
``tile_arange()``, and ``tile_full()``. Keep kernels in this module on the
standard 64-thread block size so it does not acquire extra block-dimension
variants.
"""

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_DIM = 64
TILE_M = wp.constant(8)


@wp.kernel
def test_tile_tile_preserve_type_kernel(x: wp.array[Any], y: wp.array[Any]):
    a = x[0]
    t = wp.tile(a, preserve_type=True)
    wp.tile_store(y, t)


wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.quat], "y": wp.array[wp.quat]})
wp.overload(test_tile_tile_preserve_type_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


@wp.kernel
def test_tile_tile_scalar_expansion_kernel(x: wp.array[float], y: wp.array[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_vec_expansion_kernel(x: wp.array[wp.vec3], y: wp.array2d[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


@wp.kernel
def test_tile_tile_mat_expansion_kernel(x: wp.array[wp.mat33], y: wp.array3d[float]):
    a = x[0]
    t = wp.tile(a)
    wp.tile_store(y, t)


def test_tile_preserves_and_expands_value_types(test, device):
    """Preserve and expand value types through ``tile()`` operations."""

    def test_func_preserve_type(type: Any):
        x = wp.ones(1, dtype=type, requires_grad=True, device=device)
        y = wp.zeros((TILE_DIM), dtype=type, requires_grad=True, device=device)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_tile_tile_preserve_type_kernel,
                dim=[TILE_DIM],
                inputs=[x],
                outputs=[y],
                block_dim=TILE_DIM,
                device=device,
            )

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), wp.full((TILE_DIM), type(1.0), dtype=type, device="cpu").numpy())
        assert_np_equal(x.grad.numpy(), wp.full((1,), type(TILE_DIM), dtype=type, device="cpu").numpy())

    test_func_preserve_type(float)
    test_func_preserve_type(wp.vec3)
    test_func_preserve_type(wp.quat)
    test_func_preserve_type(wp.mat33)

    # scalar expansion
    x = wp.ones(1, dtype=float, requires_grad=True, device=device)
    y = wp.zeros((TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_scalar_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=float, device="cpu").numpy())

    # vec expansion
    x = wp.ones(1, dtype=wp.vec3, requires_grad=True, device=device)
    y = wp.zeros((3, TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_vec_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((3, TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=wp.vec3, device="cpu").numpy())

    # mat expansion
    x = wp.ones(1, dtype=wp.mat33, requires_grad=True, device=device)
    y = wp.zeros((3, 3, TILE_DIM), dtype=float, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(
            test_tile_tile_mat_expansion_kernel,
            dim=[TILE_DIM],
            inputs=[x],
            outputs=[y],
            block_dim=TILE_DIM,
            device=device,
        )

    y.grad = wp.ones_like(y)

    tape.backward()

    assert_np_equal(y.numpy(), wp.full((3, 3, TILE_DIM), 1.0, dtype=float, device="cpu").numpy())
    assert_np_equal(x.grad.numpy(), wp.full((1,), wp.float32(TILE_DIM), dtype=wp.mat33, device="cpu").numpy())


@wp.kernel
def test_tile_untile_preserve_type_kernel(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a, preserve_type=True)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.quat], "y": wp.array[wp.quat]})
wp.overload(test_tile_untile_preserve_type_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


@wp.kernel
def test_tile_untile_kernel(x: wp.array[Any], y: wp.array[Any]):
    i = wp.tid()
    a = x[i]
    t = wp.tile(a)
    b = wp.untile(t)
    y[i] = b


wp.overload(test_tile_untile_kernel, {"x": wp.array[float], "y": wp.array[float]})
wp.overload(test_tile_untile_kernel, {"x": wp.array[wp.vec3], "y": wp.array[wp.vec3]})
wp.overload(test_tile_untile_kernel, {"x": wp.array[wp.mat33], "y": wp.array[wp.mat33]})


def test_tile_untile(test, device):
    """Preserve values, types, and gradients through ``tile()`` and ``untile()``."""

    def test_func_preserve_type(type: Any):
        x = wp.ones(TILE_DIM, dtype=type, requires_grad=True, device=device)
        y = wp.zeros_like(x)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_tile_untile_preserve_type_kernel,
                dim=TILE_DIM,
                inputs=[x],
                outputs=[y],
                block_dim=TILE_DIM,
                device=device,
            )

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), x.numpy())
        assert_np_equal(x.grad.numpy(), wp.ones_like(x).numpy())

    test_func_preserve_type(float)
    test_func_preserve_type(wp.vec3)
    test_func_preserve_type(wp.quat)
    test_func_preserve_type(wp.mat33)

    def test_func(type: Any):
        x = wp.ones(TILE_DIM, dtype=type, requires_grad=True, device=device)
        y = wp.zeros_like(x)

        tape = wp.Tape()
        with tape:
            wp.launch(test_tile_untile_kernel, dim=TILE_DIM, inputs=[x], outputs=[y], block_dim=TILE_DIM, device=device)

        y.grad = wp.ones_like(y)

        tape.backward()

        assert_np_equal(y.numpy(), x.numpy())
        assert_np_equal(x.grad.numpy(), wp.ones_like(x).numpy())

    test_func(float)
    test_func(wp.vec3)
    test_func(wp.mat33)


@wp.kernel
def tile_untile_scalar_kernel(output: wp.array[int]):
    i = wp.tid()
    t = wp.tile(i) * 2
    s = wp.untile(t)
    output[i] = s


def test_tile_untile_scalar(test, device):
    """Convert scalar thread indices through ``tile()`` and ``untile()`` on an unaligned grid."""

    # use an unaligned grid dimension
    N = TILE_DIM * 4 + 5

    output = wp.zeros(shape=N, dtype=int, requires_grad=True, device=device)

    with wp.Tape():
        wp.launch(tile_untile_scalar_kernel, dim=N, inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy(), np.arange(N) * 2)


@wp.kernel
def test_untile_vector_kernel(input: wp.array[wp.vec3], output: wp.array[wp.vec3]):
    i = wp.tid()

    v = input[i] * 0.5

    t = wp.tile(v)
    u = wp.untile(t)

    output[i] = u * 2.0


def test_tile_untile_vector(test, device):
    """Preserve vector values and gradients through ``tile()`` and ``untile()``."""

    input = wp.full(TILE_DIM, wp.vec3(1.0, 2.0, 3.0), requires_grad=True, device=device)
    output = wp.zeros_like(input, device=device)

    with wp.Tape() as tape:
        wp.launch(test_untile_vector_kernel, dim=TILE_DIM, inputs=[input, output], block_dim=TILE_DIM, device=device)

    output.grad = wp.ones_like(output, device=device)
    tape.backward()

    assert_np_equal(output.numpy(), input.numpy())
    assert_np_equal(input.grad.numpy(), np.ones((TILE_DIM, 3)))


@wp.struct
class TestStruct:
    x: wp.float32
    y: wp.vec3


@wp.struct
class TestStructWithArray:
    """Struct with array field for testing tile_zeros with complex types."""

    x: wp.array[wp.float64]


@wp.kernel
def test_tile_construction_kernel(
    out_zeros: wp.array[float],
    out_ones: wp.array[float],
    out_arange: wp.array[float],
    out_full_twos: wp.array[float],
    out_full_vecs: wp.array[wp.vec3],
    out_full_mats: wp.array[wp.mat33],
    out_full_structs_register: wp.array[TestStruct],
    out_full_structs_shared: wp.array[TestStruct],
    out_zeros_struct_with_array: wp.array[TestStructWithArray],
):
    zeros = wp.tile_zeros(TILE_M, dtype=float)
    ones = wp.tile_ones(TILE_M, dtype=float)
    arange = wp.tile_arange(TILE_M, dtype=float)
    full_twos = wp.tile_full(TILE_M, value=2.0, dtype=float)
    full_vecs = wp.tile_full(TILE_M, value=wp.vec3(1.0), dtype=wp.vec3)
    full_mats = wp.tile_full(TILE_M, value=wp.mat33(1.0), dtype=wp.mat33)

    ts = TestStruct()
    ts.x = wp.float32(2.0)
    ts.y = wp.vec3(1.0)
    full_structs_register = wp.tile_full(TILE_M, value=ts, dtype=TestStruct, storage="register")
    full_structs_shared = wp.tile_full(TILE_M, value=ts, dtype=TestStruct, storage="shared")

    zeros_struct_with_array = wp.tile_zeros(TILE_M, dtype=TestStructWithArray)

    wp.tile_store(out_zeros, zeros)
    wp.tile_store(out_ones, ones)
    wp.tile_store(out_arange, arange)
    wp.tile_store(out_full_twos, full_twos)
    wp.tile_store(out_full_vecs, full_vecs)
    wp.tile_store(out_full_mats, full_mats)
    wp.tile_store(out_full_structs_register, full_structs_register)
    wp.tile_store(out_full_structs_shared, full_structs_shared)
    wp.tile_store(out_zeros_struct_with_array, zeros_struct_with_array)


def test_tile_construction(test, device):
    """Construct tiles with initialization helpers and composite types."""

    zeros = wp.empty(TILE_M, dtype=float, device=device)
    ones = wp.empty(TILE_M, dtype=float, device=device)
    arange = wp.empty(TILE_M, dtype=float, device=device)
    full_twos = wp.empty(TILE_M, dtype=float, device=device)
    full_vecs = wp.empty(TILE_M, dtype=wp.vec3, device=device)
    full_mats = wp.empty(TILE_M, dtype=wp.mat33, device=device)
    full_structs_register = wp.empty(TILE_M, dtype=TestStruct, device=device)
    full_structs_shared = wp.empty(TILE_M, dtype=TestStruct, device=device)
    zeros_struct_with_array = wp.empty(TILE_M, dtype=TestStructWithArray, device=device)

    wp.launch_tiled(
        test_tile_construction_kernel,
        dim=1,
        inputs=[],
        outputs=[
            zeros,
            ones,
            arange,
            full_twos,
            full_vecs,
            full_mats,
            full_structs_register,
            full_structs_shared,
            zeros_struct_with_array,
        ],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(zeros.numpy(), np.zeros(TILE_M, dtype=float))
    assert_np_equal(ones.numpy(), np.ones(TILE_M, dtype=float))
    assert_np_equal(full_twos.numpy(), np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_vecs.numpy(), np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(full_mats.numpy(), np.ones((TILE_M, 3, 3), dtype=float))
    assert_np_equal(full_structs_register.numpy()["x"], np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_structs_register.numpy()["y"], np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(full_structs_shared.numpy()["x"], np.full(TILE_M, 2.0, dtype=float))
    assert_np_equal(full_structs_shared.numpy()["y"], np.ones((TILE_M, 3), dtype=float))
    assert_np_equal(arange.numpy(), np.arange(TILE_M, dtype=float))

    # Verify struct with array field is zero-initialized
    # The array field is an array_t with (data, grad, shape, strides, ndim) - all should be zero
    struct_arr_np = zeros_struct_with_array.numpy()
    test.assertTrue(np.all(struct_arr_np["x"]["data"] == 0))
    test.assertTrue(np.all(struct_arr_np["x"]["grad"] == 0))
    test.assertTrue(np.all(struct_arr_np["x"]["ndim"] == 0))


@wp.kernel
def tile_ones_kernel(out: wp.array[float]):
    i = wp.tid()

    t = wp.tile_ones(dtype=float, shape=(16, 16))
    s = wp.tile_sum(t)

    wp.tile_store(out, s)


def test_tile_ones(test, device):
    """Fill a tile with ones and reduce it to the expected sum."""

    output = wp.zeros(1, dtype=float, device=device)

    with wp.Tape():
        wp.launch_tiled(tile_ones_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    test.assertAlmostEqual(output.numpy()[0], 256.0)


@wp.kernel
def tile_arange_kernel(out: wp.array2d[int]):
    i = wp.tid()

    a = wp.tile_arange(17, dtype=int)
    b = wp.tile_arange(5, 22, dtype=int)
    c = wp.tile_arange(0, 34, 2, dtype=int)
    d = wp.tile_arange(-1, 16, dtype=int)
    e = wp.tile_arange(17, 0, -1, dtype=int)

    wp.tile_store(out[0], a)
    wp.tile_store(out[1], b)
    wp.tile_store(out[2], c)
    wp.tile_store(out[3], d)
    wp.tile_store(out[4], e)


def test_tile_arange_start_stop_step_forms(test, device):
    """Construct integer tile ranges with supported start, stop, and step forms."""

    N = 17

    output = wp.zeros(shape=(5, N), dtype=int, device=device)

    with wp.Tape():
        wp.launch_tiled(tile_arange_kernel, dim=[1], inputs=[output], block_dim=TILE_DIM, device=device)

    assert_np_equal(output.numpy()[0], np.arange(17))
    assert_np_equal(output.numpy()[1], np.arange(5, 22))
    assert_np_equal(output.numpy()[2], np.arange(0, 34, 2))
    assert_np_equal(output.numpy()[3], np.arange(-1, 16))
    assert_np_equal(output.numpy()[4], np.arange(17, 0, -1))


class TestTileConstruction(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestTileConstruction,
    "test_tile_preserves_and_expands_value_types",
    test_tile_preserves_and_expands_value_types,
    devices=get_cuda_test_devices(),
)
add_function_test(TestTileConstruction, "test_tile_untile", test_tile_untile, devices=devices)
add_function_test(TestTileConstruction, "test_tile_untile_scalar", test_tile_untile_scalar, devices=devices)
add_function_test(TestTileConstruction, "test_tile_untile_vector", test_tile_untile_vector, devices=devices)
add_function_test(TestTileConstruction, "test_tile_construction", test_tile_construction, devices=devices)
add_function_test(TestTileConstruction, "test_tile_ones", test_tile_ones, devices=devices)
add_function_test(
    TestTileConstruction,
    "test_tile_arange_start_stop_step_forms",
    test_tile_arange_start_stop_step_forms,
    devices=devices,
)
add_function_test(
    TestTileConstruction,
    "test_tile_untile_cpu_blocks",
    test_tile_untile,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileConstruction,
    "test_tile_construction_cpu_blocks",
    test_tile_construction,
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
