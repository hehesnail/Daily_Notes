import d2ltvm
import numpy as np
import tvm
from tvm import te
import timeit
import os
os.environ['KMP_AFFINITY']='granularity=fine,noduplicates,compact,1,0'

target = 'llvm'

def depthwise_conv_pack(c, nh, nw, kh, kw, ph, pw, tc):
    """Pack data and weight for depthwise convolution
       Note that the input channel of kernel is specified as 1,
       and the output channel of kernel equals the input channel of data

    c : input channel of data and output channel of kernel
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    tc : the tiling size of channels
    """
    X = te.placeholder((c, nh, nw), name='x')
    K = te.placeholder((c, 1, kh, kw), name='k')
    PaddedX = d2ltvm.padding(X, ph, pw) if ph*pw != 0 else X
    # make sure the channel tiling is valid
    if c < tc:
        tc = c
    assert c % tc == 0
    # pack X and K
    PackedX = te.compute(
        (c // tc, nh+ph*2, nw+pw*2, tc), 
        lambda c_out, x, y, c_in: PaddedX[c_out*tc+c_in, x, y],
        name='PackedX')
    PackedK = te.compte(
        (c // tc, 1, kh, kw, 1, tc), 
        lambda c_out, _, x, y, __, c_in: K[c_out*tc+c_in, 0, x, y], 
        name='PackedK')

    return X, K, PaddedX, PackedX, PackedK

def depthwise_conv(c, nh, nw, kh, kw, ph, pw, sh, sw, tc):
    """depthwise conv

    c : number of channels for both input and output.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    tc : the tiling sizes of channels
    """
    X, K, PaddedX, PackedX, PackedK = depthwise_conv_pack(
        c, nh, nw, kh, kw, ph, pw, tc)
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = d2ltvm.conv_out_size(nh, kh, ph, sh)
    ow = d2ltvm.conv_out_size(nw, kw, pw, sw)
    # compute Y in packed layout
    PackedY = te.compute(
        (c//tc, oh, ow, tc), 
        lambda c_out, x, y, c_in: te.sum(
            (PackedX[c_out*tc+c_in, x*sh+rkh, y*sw+rkw, c_in] *
             PackedK[c_out*tc+c_in, 0, rkh, rkw, 0, c_in]), 
            axis=[rkh, rkw]), name='PackedY')
    # Unpack the results
    Y = te.compute(
        (c, oh, ow), 
        lambda c, x, y: PackedY[c//tc, x, y, c%tc],
        name='Y')
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY

c, n, k, p, s, tc = 32, 64, 3, 1, 1, 16

X, K, Y, _, _, _, _ = depthwise_conv(c, n, n, k, k, p, p, s, s, tc)
mod = tvm.build(te.create_schedule(Y.op), [X, K, Y])

data, weight, out = d2ltvm.get_conv_data(c, c, n, k, p, s, tvm.nd.array, conv_type='depthwise')
mod(data, weight, out)

data, weight, bias, out_mx = d2ltvm.get_conv_data_mxnet(c, c, n, k, p, s, conv_type='depthwise')
d2ltvm.depthwise_conv_mxnet(data, weight, bias, out_mx, k, p, s)
np.testing.assert_allclose(out_mx[0].asnumpy(), out.asnumpy(), atol=1e-5)

tc, tw = 16, 4

def depthwise_cached_block(c, n, k, p, s):
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = depthwise_conv(
        c, n, n, k, k, p, p, s, s, tc)
    sch = te.create_schedule(Y.op)

    CachedY = sch.cache_write(PackedY, 'global')
    c_out, h, w, c_in = sch[PackedY].op.axis
    w_out, w_in = sch[PackedY].split(w, factor=tw)
    sch[PackedY].reorder(c_out, h, w_out, w_in, c_in)
    c_out_h = sch[PackedY].fuse(c_out, h)
    sch[PackedY].parallel(c_out_h)
    sch[CachedY].compute_at(sch[PackedY], w_out)

    cc_out, ch, cw, cc_in = sch[CachedY].op.axis
    kh, kw = sch[CachedY].op.reduce_axis
    sch[CachedY].reorder(cc_out, ch, kh, kw, cw, cc_in)
    sch[CachedY].vectorize(cc_in)
    sch[CachedY].unroll(cw)

    # schedule the padding by thread-level parallelism
    if PaddedX != X:
        sch[PaddedX].parallel(PaddedX.op.axis[0])
    # optimize packing X and K
    sch[PackedX].parallel(sch[PackedX].fuse(*PackedX.op.axis[0:2]))
    sch[PackedX].unroll(PackedX.op.axis[-1])
    sch[PackedK].parallel(sch[PackedK].fuse(*PackedK.op.axis[0:2]))
    sch[PackedK].unroll(PackedK.op.axis[-1])
    #optimize the unpacking Y
    sch[Y].parallel(sch[Y].fuse(*Y.op.axis[0:2]))
    sch[Y].unroll(Y.op.axis[-1])

    return sch, (X, K, Y)

# c, n, k, p, s were defined in the previous code block
sch, args = depthwise_cached_block(c, n, k, p, s)
# Uncomment the following line to see the long
# psuedo codes because of unrolling.
print(tvm.lower(sch, args, simple_mode=True))