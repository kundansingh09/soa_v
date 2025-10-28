"""
Main script to evaluate the trained backdoored VSL models.

This script:
1. Loads the saved backdoored models (A, B, and Server).
2. Loads the CIFAR-10 training and test datasets.
3. Re-computes the non-augmented trigger by first passing the
   entire training set through the backdoored model A.
4. Calculates CDA (Clean Data Accuracy) on the clean test set.
5. Calculates ASR (Attack Success Rate) by applying the
   non-augmented trigger to non-target samples in the test set.
6. Prints the final results.

Note: LIA (Label Inference Accuracy) is calculated *during*
training (in run_villain_attack.py) and is not computed here.
"""

import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

# --- 1. Add Project Root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Project-specific Imports ---
try:
    from config.cifar10 import (
        BATCH_SIZE, SPLIT_METHOD, PARTICIPANT_MODEL_TYPE,
        NUM_PARTICIPANTS, EMBEDDING_AGGREGATION, TRIGGER_MAGNITUDE_BETA
    )
    from data.cifar10_loader import load_cifar10, get_vsl_dataloaders
    from src.models.participant_vgg import ParticipantVGG16
    from src.models.server_model import ServerModel
    from src.attack.data_poisoning import create_trigger_mask, fabricate_trigger
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Details: {e}")
    print(f"Please ensure all modules are correctly defined.")
    sys.exit(1)

# --- 3. Configuration ---
# Must match the target label used in run_villain_attack.py
TARGET_LABEL = 3 

MODEL_SAVE_DIR = os.path.join(project_root, "results", "models")
ATTACK_A_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_A.pth")
ATTACK_B_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_B.pth")
ATTACK_S_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_S.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_non_augmented_trigger(model_A: nn.Module, 
                              train_loader_A: torch.utils.data.DataLoader,
                              embed_dim: int,
                              beta: float) -> torch.Tensor:
    """
    Re-calculates the non-augmented trigger by analyzing all
    training set embeddings from the attacker's model.
    """
    print("Calculating trigger: Passing full training set through model A...")
    all_embeddings = []
    
    for data_A in tqdm(train_loader_A, desc="Trigger Calc"):
        data_A = data_A.to(DEVICE)
        e_A = model_A(data_A)
        all_embeddings.append(e_A.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Collected {all_embeddings.shape[0]} training embeddings.")

    # 1. Create Trigger Mask
    # Assume mask size m is 10% of embedding dim
    m = int(embed_dim * 0.1) 
    mask = create_trigger_mask(all_embeddings, m=m)

    # 2. Fabricate Trigger (Non-Augmented)
    std_dev_all = torch.std(all_embeddings, dim=0)
    trigger = fabricate_trigger(mask, std_dev_all, beta=beta)
    
    print("Non-augmented trigger created successfully.")
    return trigger.to(DEVICE)

@torch.no_grad()
def calculate_cda(model_A: nn.Module,
                  model_B: nn.Module,
                  model_S: nn.Module,
                  loader_A: torch.utils.data.DataLoader,
                  loader_B: torch.utils.data.DataLoader,
                  loader_S: torch.utils.data.DataLoader) -> float:
    """
    Calculates Clean Data Accuracy (CDA) on the test set.
    """
    model_A.eval()
    model_B.eval()
    model_S.eval()
    
    correct = 0
    total = 0
    
    # Note: loader_S yields (labels, global_indices)
    batch_iterator = zip(loader_A, loader_B, loader_S)
    
    for data_A, data_B, (labels_S, _) in tqdm(batch_iterator, desc="Calculating CDA"):
        data_A, data_B = data_A.to(DEVICE), data_B.to(DEVICE)
        labels_S = labels_S.to(DEVICE)
        
        # Benign forward pass
        e_A = model_A(data_A)
        e_B = model_B(data_B)
        
        e_agg = torch.cat((e_A, e_B), dim=1)
        logits = model_S(e_agg)
        
        _, predicted = torch.max(logits.data, 1)
        total += labels_S.size(0)
        correct += (predicted == labels_S).sum().item()
        
    return 100.0 * correct / total

@torch.no_grad()
def calculate_asr(model_A: nn.Module,
                  model_B: nn.Module,
                  model_S: nn.Module,
                  loader_A: torch.utils.data.DataLoader,
                  loader_B: torch.utils.data.DataLoader,
                  loader_S: torch.utils.data.DataLoader,
                  trigger: torch.Tensor,
                  target_label: int) -> float:
    """
    Calculates Attack Success Rate (ASR) on the test set.
    """
    model_A.eval()
    model_B.eval()
    model_S.eval()
    
    misclassified_as_target = 0
    total_non_target = 0
    
    batch_iterator = zip(loader_A, loader_B, loader_S)
    
    for data_A, data_B, (labels_S, _) in tqdm(batch_iterator, desc="Calculating ASR"):
        data_A, data_B = data_A.to(DEVICE), data_B.to(DEVICE)
        labels_S = labels_S.to(DEVICE)
        
        # 1. Filter out samples that *already* belong to the target class
        non_target_mask = (labels_S != target_label)
        if non_target_mask.sum() == 0:
            continue
            
        data_A_non_target = data_A[non_target_mask]
        data_B_non_target = data_B[non_target_mask]
        labels_S_non_target = labels_S[non_target_mask]
        
        total_non_target += labels_S_non_target.size(0)
        
        # 2. Perform triggered forward pass
        e_A = model_A(data_A_non_target)
        e_B = model_B(data_B_non_target)
        
        # Apply the additive trigger
        e_A_triggered = e_A + trigger
        
        e_agg = torch.cat((e_A_triggered, e_B), dim=1)
        logits = model_S(e_agg)
        
        _, predicted = torch.max(logits.data, 1)
        
        # 3. Check if they were misclassified *into the target label*
        misclassified_as_target += (predicted == target_label).sum().item()
        
    return 100.0 * misclassified_as_target / total_non_target

def main():
    """
    Main function to orchestrate the evaluation.
    """
    print("--- 🚀 Starting Evaluation Script ---")
    
    # --- 1. Load Models ---
    print("Loading backdoored models...")
    try:
        model_A = ParticipantVGG16().to(DEVICE)
        model_B = ParticipantVGG16().to(DEVICE)
        model_S = ServerModel(
            participant_embedding_dim=model_A.embedding_dim,
            num_participants=NUM_PARTICIPANTS,
            num_classes=10
        ).to(DEVICE)
        
        model_A.load_state_dict(torch.load(ATTACK_A_PATH, map_location=DEVICE))
        model_B.load_state_dict(torch.load(ATTACK_B_PATH, map_location=DEVICE))
        model_S.load_state_dict(torch.load(ATTACK_S_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"Error: Could not find model file. {e}")
        print("Please run 'run_villain_attack.py' first.")
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Models loaded successfully.")
    
    # --- 2. Load Data ---
    print("Loading CIFAR-10 data (train and test sets)...")
    trainset, testset = load_cifar10()
    
    # Need train loader for attacker A to re-calculate trigger
    train_loader_A, _, _ = get_vsl_dataloaders(
        original_dataset=trainset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=False # No shuffle needed for stats
    )
    
    # Need test loaders for evaluation
    test_loader_A, test_loader_B, test_loader_S = get_vsl_dataloaders(
        original_dataset=testset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=False
    )
    
    # --- 3. Get Trigger ---
    trigger = get_non_augmented_trigger(
        model_A, 
        train_loader_A,
        embed_dim=model_A.embedding_dim,
        beta=TRIGGER_MAGNITUDE_BETA
    )
    
    # --- 4. Calculate Metrics ---
    cda = calculate_cda(
        model_A, model_B, model_S,
        test_loader_A, test_loader_B, test_loader_S
    )
    
    asr = calculate_asr(
        model_A, model_B, model_S,
        test_loader_A, test_loader_B, test_loader_S,
        trigger, TARGET_LABEL
    )
    
    # --- 5. Print Results ---
    print("\n" + "="*30)
    print("--- 📊 Evaluation Results ---")
    print("="*30)
    print(f"Clean Data Accuracy (CDA): {cda:.2f}%")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print(f"Label Inference Acc. (LIA): (Calculated during training)")
    print("\n--- ✅ Evaluation Script Finished ---")


if __name__ == "__main__":
    main()