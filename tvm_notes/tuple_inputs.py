from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

n = te.var('n')
m = te.var('m')
A0 = te.placeholder((m, n), name='A0')
A1 = te.placeholder((m, n), name='A1')

B0, B1 = te.compute((m, n), lambda i, j: (A0[i,j]+2, A1[i,j]*3), name='B')
s = te.create_schedule(B0.op)
print(tvm.lower(s, [A0, A1, B0, B1], simple_mode=True))

# x and y are the operands of reduction, both of them is a tuple of index
# and value.
def fcombine(x, y):
    lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs


# our identity element also need to be a tuple, so `fidentity` accepts
# two types as inputs.
def fidentity(t0, t1):
    return tvm.tir.const(-1, t0), tvm.te.min_value(t1)


argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

# describe the reduction computation
m = te.var("m")
n = te.var("n")
idx = te.placeholder((m, n), name="idx", dtype="int32")
val = te.placeholder((m, n), name="val", dtype="int32")
k = te.reduce_axis((0, n), "k")
T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")

# the generated IR code would be:
s = te.create_schedule(T0.op)
print(tvm.lower(s, [idx, val, T0, T1], simple_mode=True))