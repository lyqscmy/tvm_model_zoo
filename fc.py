import nnvm.symbol as sym
import tvm
import numpy as np
from functools import reduce
from operator import mul


def fc_layer(data, units, name):
    w = sym.Variable(name + "_w")
    b = sym.Variable(name + "_b")
    fc = sym.dense(data=data, weight=w, bias=b, units=units, name=name + '_fc')
    relu = sym.relu(data=fc, name=name + '_relu')

    gamma = sym.Variable(name + "_gamma")
    beta = sym.Variable(name + "_beta")
    moving_mean = sym.Variable(name + "_moving_mean")
    moving_var = sym.Variable(name + "_moving_var")
    bn = sym.batch_norm(
        data=relu,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        name=name + '_bn')
    return bn


def deep(layers, units):
    data = sym.Variable("data")

    deep = fc_layer(data, units[0], "fc_layer_1")
    for i in range(layers - 1):
        deep = fc_layer(deep, units[i + 1], "fc_layer_" + str(i + 2))
    return deep


layers = 1
units = [3] * layers
net = deep(layers, units)
batch_size = 2
input_shape = (batch_size, 3)
output_shape = (batch_size, units[-1])

params = {}
for i in range(layers):
    w_shape = (units[i], input_shape[1])
    w_np = np.ones(reduce(mul, w_shape), dtype="float32").reshape(w_shape)
    b_np = np.ones(units[i]).astype("float32")
    params["fc_layer_" + str(i + 1) + "_w"] = tvm.ndarray.array(w_np)
    params["fc_layer_" + str(i + 1) + "_b"] = tvm.ndarray.array(b_np)

    gamma_np = np.ones(units[i], dtype="float32")
    beta_np = np.zeros(units[i], dtype="float32")
    params["fc_layer_" + str(i + 1) + "_gamma"] = tvm.ndarray.array(gamma_np)
    params["fc_layer_" + str(i + 1) + "_beta"] = tvm.ndarray.array(beta_np)
    params["fc_layer_" + str(i + 1) +
           "_moving_mean"] = tvm.ndarray.array(beta_np)
    params["fc_layer_" + str(i + 1) +
           "_moving_var"] = tvm.ndarray.array(gamma_np)
