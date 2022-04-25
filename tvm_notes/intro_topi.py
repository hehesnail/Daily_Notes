from __future__ import absolute_import, print_function
import tvm
import tvm.testing
from tvm import te
from tvm import topi
import numpy as np

# Not use topi, we need to define the reduce axis
n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='a')
k = te.reduce_axis((0, m), name='k')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')
s = te.create_schedule(B.op)
print(tvm.lower(s, [A], simple_mode=True))

# use topi, quite like the numpy.sum
C = topi.sum(A, axis=1)
ts = te.create_schedule(C.op)
print(tvm.lower(ts, [A], simple_mode=True))

# operator overloading
x, y = 100, 100
a = te.placeholder((x, y, y), name='a')
b = te.placeholder((y, y), name='b')
c = a + b
d = a * b

# generic schedules
e = topi.elemwise_sum([c, d])
f = e / 2.0
g = topi.sum(f)
with tvm.target.cuda():
    sg = topi.cuda.schedule_reduce(g)
    print(tvm.lower(sg, [a, b], simple_mode=True))

# test correct
func = tvm.build(sg, [a, b, g], "cuda")
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(x, y, y)).astype(a.dtype)
b_np = np.random.uniform(size=(y, y)).astype(b.dtype)
g_np = np.sum(np.add(a_np + b_np, a_np * b_np) / 2.0)
a_nd = tvm.nd.array(a_np, ctx)
b_nd = tvm.nd.array(b_np, ctx)
g_nd = tvm.nd.array(np.zeros(g_np.shape, dtype=g_np.dtype), ctx)
func(a_nd, b_nd, g_nd)
tvm.testing.assert_allclose(g_nd.asnumpy(), g_np, rtol=1e-5)

# softmax
tarray = te.placeholder((512, 512), name='tarray')
softmax_topi = topi.nn.softmax(tarray)
with tvm.target.Target("cuda"):
    sst = topi.cuda.schedule_softmax(softmax_topi)
    print(tvm.lower(sst, [tarray], simple_mode=True))

with tvm.target.Target("x86"):
    sst = topi.x86.schedule_softmax(softmax_topi)
    print(tvm.lower(sst, [tarray], simple_mode=True))    

# conv + relu
data = te.placeholder((1, 3, 224, 224))
kernel = te.placeholder((10, 3, 5, 5))
with tvm.targt.Target("cuda"):
    conv = topi.cuda.conv2d_nchw(data, kernel, 1, 2, 1)
    out = topi.nn.relu(conv)
    sconv = topi.cuda.schedule_conv2d_nchw([out])
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))
with tvm.target.Target("x86"):
    conv = topi.x86.conv2d_nchw(data, kernel, 1, 2, 1)
    out = topi.nn.relu(conv)
    sconv = topi.x86.schedule_conv2d_nchw([out])
    print(tvm.lower(sconv, [data, kernel], simple_mode=True))

