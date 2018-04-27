from fc import net, input_shape, params
import nnvm
from pathlib import Path

compute_graph = nnvm.graph.create(net)
print("-------compute graph-------")
print(compute_graph.ir())

deploy_graph, lib, params = nnvm.compiler.build(
    compute_graph, target="llvm", shape={"data": input_shape}, params=params)

print("-------deploy graph-------")
print(deploy_graph.ir())
print("-----optimized params-----")
print(params)

current_path = Path('.')
path_lib = current_path / 'deploy.so'
lib.export_library(str(path_lib.absolute()))
with open("deploy.json", "w") as fo:
    fo.write(deploy_graph.json())
with open("deploy.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
