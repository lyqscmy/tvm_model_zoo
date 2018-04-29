import nnvm.symbol as sym
import tvm
import numpy as np
from functools import reduce
from operator import mul


def fc_layer(data, units, name):
    w = sym.Variable(name + "_fc_weight")
    b = sym.Variable(name + "_fc_bias")
    fc = sym.dense(data=data, weight=w, bias=b, units=units, name=name + '_fc')

    gamma = sym.Variable(name + "_bn_gamma")
    beta = sym.Variable(name + "_bn_beta")
    moving_mean = sym.Variable(name + "_bn_moving_mean")
    moving_var = sym.Variable(name + "_bn_moving_var")

    bn = sym.batch_norm(
        data=fc,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        name=name + '_bn')

    relu = sym.relu(data=bn, name=name + '_relu')
    return relu


def mlp(units):
    data = sym.Variable("data")

    deep = fc_layer(data, units[0], "fc_layer1")
    deep = fc_layer(deep, units[1], "fc_layer2")

    name = "output_layer"
    w = sym.Variable(name+"_fc_weight")
    b = sym.Variable(name+"_fc_bias")
    fc = sym.dense(data=deep, weight=w, bias=b,
                   units=units[2], name=name+"_fc")

    gamma = sym.Variable(name + "_bn_gamma")
    beta = sym.Variable(name + "_bn_beta")
    moving_mean = sym.Variable(name + "_bn_moving_mean")
    moving_var = sym.Variable(name + "_bn_moving_var")

    bn = sym.batch_norm(
        data=fc,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        name=name + '_bn')

    mlp = sym.softmax(data=bn, name=name+'softmax')
    return mlp


units = [1024, 1024, 2]
net = mlp(units)

batch_size = 1000
input_shape = (batch_size, 1024)
output_shape = (batch_size, 2)

params = {}

# layer 1 params
name = "fc_layer1"
w_shape = (units[0], input_shape[1])
w_np = np.ones(reduce(mul, w_shape), dtype="float32").reshape(w_shape)
b_np = np.ones(units[0]).astype("float32")
params[name+"_fc_weight"] = tvm.ndarray.array(w_np)
params[name+"_fc_bias"] = tvm.ndarray.array(b_np)

gamma_np = np.ones(units[0], dtype="float32")
beta_np = np.zeros(units[0], dtype="float32")
params[name+"_bn_gamma"] = tvm.ndarray.array(gamma_np)
params[name+"_bn_beta"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_mean"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_var"] = tvm.ndarray.array(gamma_np)

# layer 2 params
name = "fc_layer2"
w_shape = (units[1], input_shape[1])
w_np = np.ones(reduce(mul, w_shape), dtype="float32").reshape(w_shape)
b_np = np.ones(units[1]).astype("float32")
params[name+"_fc_weight"] = tvm.ndarray.array(w_np)
params[name+"_fc_bias"] = tvm.ndarray.array(b_np)

gamma_np = np.ones(units[1], dtype="float32")
beta_np = np.zeros(units[1], dtype="float32")
params[name+"_bn_gamma"] = tvm.ndarray.array(gamma_np)
params[name+"_bn_beta"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_mean"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_var"] = tvm.ndarray.array(gamma_np)


# layer 3 params
name = "output_layer"
w_shape = (units[2], input_shape[1])
w_np = np.ones(reduce(mul, w_shape), dtype="float32").reshape(w_shape)
b_np = np.ones(units[2]).astype("float32")
params[name+"_fc_weight"] = tvm.ndarray.array(w_np)
params[name+"_fc_bias"] = tvm.ndarray.array(b_np)

gamma_np = np.ones(units[2], dtype="float32")
beta_np = np.zeros(units[2], dtype="float32")
params[name+"_bn_gamma"] = tvm.ndarray.array(gamma_np)
params[name+"_bn_beta"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_mean"] = tvm.ndarray.array(beta_np)
params[name+"_bn_moving_var"] = tvm.ndarray.array(gamma_np)


