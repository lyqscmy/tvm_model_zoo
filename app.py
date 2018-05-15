from flask import Flask, g
import tvm
from pathlib import Path
from tvm.contrib import graph_runtime
app = Flask(__name__)


def get_module():
    if 'module' not in g:
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
        g.module = module

    return g.module


@app.route("/")
def hello():
    module = get_module()
    module.run()
    return "Hello World!"
