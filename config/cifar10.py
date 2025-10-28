"""
Configuration file for the VILLAIN CIFAR-10 Replication.

This file stores all static hyperparameters as specified by the
VILLAIN paper (usenixsecurity23-bai) to ensure reproducibility.
"""

# --- 1. Dataset Parameters ---
DATASET = 'CIFAR-10'
BATCH_SIZE = 128  # [cite: 333]
# How to split the 32x32x3 image. 'vertical' = two 16x32x3 halves.
# This is an assumption based on the use of VGG16, which requires
# spatial data, to achieve the paper's high CDA.
SPLIT_METHOD = 'vertical'

# --- 2. Model Parameters ---
NUM_PARTICIPANTS = 2  # [cite: 317]
# Participant model is VGG16, Server is 3-layer FC NN [cite: 319]
PARTICIPANT_MODEL_TYPE = 'vgg16'
SERVER_MODEL_TYPE = 'fc_3_layer'
# Default aggregation is concatenation [cite: 331]
EMBEDDING_AGGREGATION = 'concatenation'

# --- 3. Training Parameters ---
# ASSUMPTION: Based on Figure 9(b) showing high ASR for LRs
# 0.03-0.2, and the paper's strategy to "increase the learning
# rate".
BENIGN_LR = 0.01
SERVER_LR = 0.01
ATTACKER_LR = 0.1  # 10x benign rate to amplify influence

# --- 4. Attack: General Parameters ---
POISONING_RATE = 0.01  # 1% [cite: 332]

# --- 5. Attack: Label Inference Parameters ---
# Number of candidates for embedding swapping per batch [cite: 333]
N_CANDIDATES = 14
# Gradient norm threshold (μ). Set dynamically in the code
# "based on the average value of the gradient L2 norm value"[cite: 488].
GRADIENT_NORM_THRESHOLD_DYNAMIC = True
# ASSUMPTION: Gradient ratio threshold (θ)[cite: 235].
# Not specified in paper. Assuming a strict value to achieve
# the paper's high LIA (96.08%).
GRADIENT_RATIO_THRESHOLD = 1.5

# --- 6. Attack: Data Poisoning Parameters ---
TRIGGER_MAGNITUDE_BETA = 0.4  # 
# Dropout for backdoor augmentation [cite: 491]
AUG_DROPOUT_RATIO = 0.75
# Shifting range for backdoor augmentation 
AUG_SHIFTING_RANGE = [0.6, 1.2]