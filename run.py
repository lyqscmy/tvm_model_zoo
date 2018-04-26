import tvm
import numpy as np
from tvm.contrib import graph_runtime
from LR import input_shape, output_shape

data = np.arange(6).reshape((input_shape)).astype("float32")
data = tvm.ndarray.array(data)
out = tvm.ndarray.empty(output_shape)

# tvm module for compiled functions.
loaded_lib = tvm.module.load("deploy.so")
# json graph
with open("deploy.json") as f:
    loaded_json = f.read()
# parameters in binary
with open("deploy.params", "rb") as f:
    loaded_params = bytearray(f.read())

ctx = tvm.cpu(0)
module = graph_runtime.create(loaded_json, loaded_lib, ctx)

module.load_params(loaded_params)

module.set_input("data", data)
module.run()
module.get_output(0, out)

print(out.asnumpy())

