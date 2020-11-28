import tvm
from tvm import te
import numpy as np

# The hybrid frontedn allow users to write preliminary versions of some 
# idioms that have been not supported by TVM officially

@tvm.te.hybrid.script
def outer_product(a, b):
    c = te.placeholder((100, 99), 'float32')
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            c[i, j] = a[i] * b[j]
    return c

a = np.random.randn(100)
b = np.random.randn(99)
c = outer_product(a, b)

a = te.placeholder((100, ), name='a')
b = te.placeholder((99, ), name='b')
c = outer_product(a, b)
i, j = c.op.axis
sch = te.create_schedule(c.op)
jo, ji = sch.split(j, 4)
sch.vectorize(ji)

print(tvm.lower(sch, [a, b, c], simple_mode=True))