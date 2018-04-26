import nnvm.symbol as sym
import numpy as np
import tvm


def LR(units):
    data = sym.Variable("data")
    w = sym.Variable("w")
    b = sym.Variable("b")
    fc = sym.dense(data=data, weight=w, bias=b, units=units, name='fc')
    return fc


net = LR(1)
input_shape = (2, 3)
output_shape = (2,)

w = np.ones(3).reshape((1, 3)).astype("float32")
b = np.array([0.5]).astype("float32")
params = {'w': tvm.ndarray.array(w), 'b': tvm.ndarray.array(b)}
