import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# tensorflow_model_server --port=8500 \
#   --rest_api_port=8501 \
#   --model_name=mnist \
#   --model_base_path=/Users/benbartkowiak/Documents/Projects/Pythons/sudoku-projekt/tensor_data/model/saved_model.pb

#   docker run -p 8501:8501 \
#   --mount type=bind,source=/Users/benbartkowiak/Documents/Projects/Pythons/sudoku-projekt/tensor_data/models/,target=/models/minst \
#   -e MODEL_NAME=minst -t tensorflow/serving

#   docker run -p 8500:8500 \
# --mount type=bind,source=/tmp2/mnist,target=/models/mnist \
# -e MODEL_NAME=mnist -t tensorflow/serving &