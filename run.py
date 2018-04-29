import timeit
import tvm
import numpy as np
from tvm.contrib import graph_runtime
from mlp import input_shape, output_shape
from pathlib import Path
from functools import reduce
from operator import mul

data = np.arange(
    reduce(mul, input_shape), dtype="float32").reshape(input_shape)
print(data)
data = tvm.ndarray.array(data)
out = tvm.ndarray.empty(output_shape)

# tvm module for compiled functions.
current_path = Path('.')
path_lib = current_path / 'deploy.so'
loaded_lib = tvm.module.load(str(path_lib.absolute()))
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
# out = out.asnumpy()

# print(out)
# out_b = tvm.ndarray.empty(output_shape)
# for i in range(10):
#     module.set_input("data", data)
#     module.run()
#     if not np.allclose(out, module.get_output(0, out_b).asnumpy()):
#         raise AssertionError('predict not stable')

def bench(data, out):
    module.set_input("data", data)
    module.run()
    module.get_output(0, out)

num_trials = 10

mlp_time = timeit.timeit(
        'bench(data, out)', number=num_trials, globals={'bench':bench, 'data': data, 'out': out})
print('='*20)
print('mlp:{:.6f}'.format(mlp_time / num_trials))
