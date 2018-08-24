import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
# load inception-v3 model defined in tensorflow-slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = "/Users/haiyong/Downloads/flower_photos/flower_processed_data.npy"
TRAIN_FILE = ""
CKPT_FILE = ""

LEARNING_RATE = 0.0001
STEPS = 300
BATCH =32
N_CLASSES = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var) 
    return variables_to_restore

def get_trainable_variables():
     

