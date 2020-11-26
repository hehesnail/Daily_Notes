import from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

m = te.var('m')
n = te.var('n')
X = te.placeholder((m, n), name='x')
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
s_scan = te.scan(s_init, s_update, s_state, inputs=[X])

s = te.create_schedule(s_scan.op)
num_thread = 256
block_X = te.thread_axis('blockIdx.x')
thread_X = te.thread_axis('threadIdx.x')
xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_X)
s[s_init].bind(xi, thread_X)
xo, xi = s[s_update].split(s.update.op.axis[1], factor=num_thread)
s[s_update].bind(xo, block_X)
s[s_update].bind(xi, thread_X)

print(tvm.lower(s, [X, s_scan], simple_mode=True))

# multi-stage scan cell
m = te.var('m')
n = te.var('n')
X = te.placeholder((m, n), name='x')
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update_s1 = te.compute((m, n), lambda t, i: s_state[t-1, i] * 2, name='s1')
s_update_s2 = te.compute((m, n), lambda t, i: s_update_s1[t, i] + X[t, i], name='s2')
s_scan = te.scan(s_init, s_update_s2, s_state, inputs=[X])

s = te.create_schedule(s_scan.op)
xo, xi = s[s_update_s2].split(s_update_s2.op.axis[1], factor=32)
s[s_update_s1].compute_at(s[s_update_s2], xo)
print(tvm.lower(s, [X, s_scan], simple_mode=True))

# multiple states
m = te.var('m')
n = te.var('n')
l = te.var('l')
X = te.placeholder((m, n), name='x')
s_state1 = te.placeholder((m, n))
s_state2 = te.placeholder((m, l))
s_init1 = te.compute((1, n), lambda _, i: X[0, i])
s_init2 = te.compute((1, l), lambda _, i: 0.0)
s_update1 = te.compute((m, n), lambda t, i: s_state1[t-1, i] + X[t, i])
s_update2 = te.compute((m, l), lambda t, i: s_state2[t-1, i] + s_state1[t-1, 0])
s_scan1, s_scan2 = te.scan(
    [s_init1, s_init2], [s_update1, s_update2], [s_state1, s_state2], inputs=[X]
)

s = te.create_schedule(s_scan1.op)
print(tvm.lower(s, [X, s_scan1, s_scan2], simple_mode=True))