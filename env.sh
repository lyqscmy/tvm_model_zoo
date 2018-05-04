source ~/venv/nnvm/bin/activate
NNVMROOT=$HOME/git/nnvm
TVMROOT=$HOME/git/nnvm/tvm
export LD_LIBRARY_PATH=$TVMROOT/lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=$TVMROOT/lib:${DYLD_LIBRARY_PATH}
export PYTHONPATH=$TVMROOT/python:$TVMROOT/topi/python:$NNVMROOT/python
