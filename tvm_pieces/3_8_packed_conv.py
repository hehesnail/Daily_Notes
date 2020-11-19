import d2ltvm
import numpy as np 
import tvm
from tvm import te
import os

c, n, tc = 4, 2, 2  # channel, height/width, and tiling size
x = np.arange(c*n*n).reshape((c, n, n)).astype('float32')
print('input shape', x.shape, '\n', x)
y = x.reshape(c//tc, n, n, tc).transpose(0, 2, 3, 1)
print('packed shape', y.shape, '\n', y)

def conv_pack(oc, ic, nh, nw, kh, kw, ph, pw, toc, tic):
    """
    Pack data and weight for conv

    oc, ic : output and input channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    toc, tic : the tiling sizes of the output and input channels
    """
    X = te.placeholder((ic, nh, nw), name='x')
    K = te.placeholder((oc, ic, kh, kw), name='k')
    PaddedX = d2ltvm.padding(X, ph, pw) if ph * pw != 0 else X
    # pack X and K
    assert ic % tic == 0 and oc % toc == 0
    PackedX = te.compute(
        (ic // tic, nh+ph*2, nw+pw*2, tic),
        lambda ic_out, x, y, ic_in: PaddedX[ic_out*tic + ic_in, x, y],
        name='PackedX' 
    )
    PackedK = te.compute(
        (oc // toc, ic // tic, kh, kw, tic, toc),
        lambda oc_out, ic_out, x, y, ic_in, oc_in: K[
            oc_out * toc + oc_in, ic_out * tic + ic_in, x, y], 
        name='PackedK'
    )

    return X, K, PaddedX, PackedX, PackedK

X, _, _, _, PackedX, _ = conv_pack(c, c, n, n, 1, 1, 0, 0, tc, tc)
mod = tvm.build(te.create_schedule(PackedX.op), [X, PackedX])
packed_x = tvm.nd.array(np.empty((c//tc, n, n, tc), dtype='float32'))
mod(tvm.nd.array(x), packed_x)
print(np.testing.assert_equal(packed_x.asnumpy(), y))

def conv(oc, ic, nh, nw, kh, kw, ph, pw, sh, sw, toc, tic):
    """2-D conv

    oc, ic : output and input channels.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    toc, tic : the tiling sizes of output channel and input channel
    """
    X, K, PaddedX, PackedX, PackedK = conv_pack(
        oc, ic, nh, nw, kh, kw, ph, pw, toc, tic)
    #reduction axes
    ric_in = te.reduce_axis((0, tic), name='ric_in')
    ric_out = te.reduce_axis((0, ic // tic), name='ric_out')
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = d2ltvm.conv_out_size(nh, kh, ph, sh)
    ow = d2ltvm.conv_out_size(nw, kw, pw, sw)
    # Computed Y in packed layput
    PackedY = te.compute(
        (oc//toc, oh, ow, toc),
        lambda oc_out, x, y, oc_in: te.sum(
            PackedX[ric_out, x*sh+rkh, y*sw+rkw, ric_in] * 
            PackedK[oc_out, ric_out, rkh, rkw, ric_in, oc_in],
            axis=[ric_out, rkh, rkw, ric_in]), name='Y')
    # Unpack the results
    Y = te.compute(
        (oc, oh, ow),
        lambda oc, x, y: PackedY[oc // toc, x, y, oc % toc],
        name='Y')
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY

oc, ic, n, k, p, s, toc, tic = 4, 6, 12, 3, 1, 1, 2, 3
X, K, Y, _, _, _, _ = conv(oc, ic, n, n, k, k, p, p, s, s, toc, tic)
mod = tvm.build(te.create_schedule(Y.op), [X, K, Y])

data, weight, out = d2ltvm.get_conv_data(oc, ic, n, k, p, s, tvm.nd.array)
mod(data, weight, out)

data, weight, bias, out_mx = d2ltvm.get_conv_data_mxnet(oc, ic, n, k, p, s)
d2ltvm.conv_mxnet(data, weight, bias, out_mx, k, p, s)
np.testing.assert_allclose(out_mx[0].asnumpy(), out.asnumpy(), atol=1e-5)

# tiling sizes for output channel, input channel, and width
toc, tic, tw = 16, 16, 4

def cached_block(oc, ic, n, k, p, s):
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = conv(
        oc, ic, n, n, k, k, p, p, s, s, toc, tic)
    s = te.create_schedule(Y.op)
    CachedY = s.cache_write(PackedY, 'local')
    oc_out, h, w, oc_in = s[PackedY].op.axis
    oc_out_h = s[PackedY].fuse(oc_out, h)
    # Parallel on the first two dimensions oc_out and h
    s[PackedY].parallel(oc_out_h)
    # optimize the  computation of a cached blocks
    w_out, w_in = s[PackedY].split(w, factor=tw)
    s[CachedY].compute_at(s[PackedY], w_out)
    print("@@@", CachedY.op.axis)
    _, _, cw, oc_in = CachedY.op.axis
    ric_out, rkh, rkw, ric_in = CachedY.op.reduce_axis
    s[CachedY].reorder(ric_out, rkh, rkw, ric_in, cw, oc_in)
    s[CachedY].unroll(ric_in)
    s[CachedY].unroll(cw)
    s[CachedY].vectorize(oc_in)
    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        s[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing X and K
    print("@@@", *PackedX.op.axis[0:2])
    print("@@@", *PackedK.op.axis[0:2])
    print("@@@", PackedX.op.axis[0:2])
    print("@@@", PackedK.op.axis[0:2])
    s[PackedX].parallel(s[PackedX].fuse(*PackedX.op.axis[0:2]))
    s[PackedX].unroll(PackedX.op.axis[-1])
    s[PackedK].parallel(s[PackedK].fuse(*PackedK.op.axis[0:2]))
    s[PackedK].unroll(PackedK.op.axis[-1])
    #optimize the unpacking Y
    s[Y].parallel(s[Y].fuse(*Y.op.axis[0:2]))
    s[Y].unroll(Y.op.axis[-1])

    return s, (X, K, Y)

s, args = cached_block(32, 32, 64, 3, 1, 1)
# Uncomment the following line to see the long
# psuedo codes because of unrolling.
print(tvm.lower(s, args, simple_mode=True))
