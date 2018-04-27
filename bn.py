import nnvm.symbol as sym
import numpy as np
import tvm

data = sym.Variable("data")
gamma = sym.Variable("gamma")
beta = sym.Variable("beta")
moving_mean = sym.Variable("moving_mean")
moving_var = sym.Variable("moving_var")
net = sym.batch_norm(
    data=data,
    gamma=gamma,
    beta=beta,
    moving_mean=moving_mean,
    moving_var=moving_var)
input_shape = (2, 3)
output_shape = input_shape
gamma_np = np.ones(input_shape[1], dtype="float32")
beta_np = np.zeros(input_shape[1], dtype="float32")
params = {
    "gamma": tvm.ndarray.array(gamma_np),
    "beta": tvm.ndarray.array(beta_np),
    "moving_mean": tvm.ndarray.array(beta_np),
    "moving_var": tvm.ndarray.array(gamma_np),
}
