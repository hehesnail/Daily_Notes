from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

@tvm.register_func("tvm.contrib.my_tvm_addone")
def my_tvm_addone(x, y):
    print("my_tvm_addone signatures: %s, %s" % (type(x), type(y)))
    tvm.nd.array(x.asnumpy() + 1).copyto(y)

A = te.placeholder((n, ), name='A')
B = te.extern(
    A.shape, 
    [A], 
    lambda ins, outs: tvm.tir.call_pakced("tvm.contrib.my_tvm_addone", ins[0], outs[0]), 
    name='C'
    )

n = 1024
ctx = tvm.cpu(0)
s = te.create_schedule(B.op)
f = tvm.build(s, [A, B], "llvm")
a = tvm.nd.array(np.random.uniform(size=(n,)).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=(n,)).astype(B.dtype), ctx)
f(a, b)
print(tvm.testing.assert_allclose(b.asnumpy(), a.asnumpy() + 1, rtol=1e-5))

