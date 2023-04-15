# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:58:32 2020

@author: stravsm
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:30 2020

@author: stravsm
"""

from tensorflow.keras.utils import plot_model

import infrastructure.generator as gen
from collections import Counter
import numpy as np
import os

import importlib
from importlib import reload
from tqdm import tqdm
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import random

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import smiles_config as sc
import infrastructure.generator as gen
import infrastructure.decoder as dec

from pathlib import Path

# Randomness is relevant in the (rare) case of using stochastic sampling
random_seed = sc.config['random_seed_global']
if random_seed != '':
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.experimental.set_seed(random_seed)

import time
from datetime import datetime
import pickle


from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import infrastructure.score as msc
import gc
import molmass



# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("decode_applied startup")

tf.get_logger().setLevel('ERROR')


eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
weights = sc.config['weights_folder'] + sc.config['weights']
picklepath = sc.config["eval_folder"] + "decode_" + pickle_id + ".pkl"
csv_path = sc.config["eval_folder"] + "decode_" + pickle_id + ".csv"
filelog_path = sc.config["eval_folder"] + "filelog_" + pickle_id

k = sc.config["eval_k"]
kk = sc.config["eval_kk"]
steps = sc.config["eval_steps"]


TRAINING_SET = sc.config['training_set']
VALIDATION_SET = sc.config['validation_set']
pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']

decoder_name = sc.config["decoder_name"]

# Should signal files be written to log progress?
output_filelog = "filelog" in sc.config
if output_filelog:
    os.mkdir(filelog_path)

# # Load dataset
# fp_db  = db.FpDatabase(sc.config['db_path'])
# fp_val = fp_db.get_grp(VALIDATION_SET)
fp_map = fpm.FingerprintMap(sc.config["fp_map"])
# fp_val = fp_val[:n_total_]

fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = False,
                                  cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()


# File for CSI:FingerID validation data
# We need to load some DB to get blueprints!
data_eval_ = sc.config["db_path_eval"]
# Load mapping table for the CSI:FingerID predictors
# Load dataset and process appropriately
db_eval = db.FpDatabase.load_from_config(data_eval_)
dataset_val = db_eval.get_grp("fold0-casmi")


pipeline_options =  db_eval.get_pipeline_options()
pipeline_options['fingerprint_selected'] = "fingerprint"

# Load dataset and sampler, apply sampler to dataset
# (so we can also evaluate from fingerprint_sampled)
fp_dataset_val_ = gen.smiles_pipeline(dataset_val, 
                                    batch_size = 1,
                                    fp_map = fp_map.positions,
                                    **pipeline_options)


round_fingerprints = True


pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']
fp_dataset_val = gen.dataset_zip(fp_dataset_val_, 
                                 pipeline_encoder, pipeline_reference,
                                 **pipeline_options)
fp_dataset_iter = iter(fp_dataset_val)
blueprints = gen.dataset_blueprint(fp_dataset_val_)

# Load models

import model

model_encode = model.EncoderModel(
                 blueprints = blueprints,
                 config = sc.config,
                 round_fingerprints = round_fingerprints)
model_decode = model.DecoderModel(
                 blueprints = blueprints,
                 config = sc.config,)
model_transcode = model.TranscoderModel(
                blueprints = blueprints,
                 config = sc.config,
                 round_fingerprints = round_fingerprints)

plot_model(model_encode, to_file='encoder_model.png', show_shapes=True)
plot_model(model_decode, to_file='decoder_model.png', show_shapes=True)
plot_model(model_transcode, to_file='transcoder_model.png', show_shapes=True)
