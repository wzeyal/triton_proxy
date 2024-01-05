sudo docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.02-py3 tritonserver --model-repository=/models



