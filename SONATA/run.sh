rcnode &

export SLICE_PATH=${PWD}/interfaces
export PYTHONPATH=${PWD}/python:$PYTHONPATH

cd controller
python3 src/controller.py etc/config

