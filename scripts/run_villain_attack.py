"""
Main script to run the VILLAIN Backdoor Attack on CIFAR-10.

This script:
1. Loads the configuration for the attack.
2. Loads the CIFAR-10 dataset and VSL data loaders.
3. Initializes one benign participant (B) and one malicious
   AttackerParticipant (A).
4. Finds one "seed sample" of the target label.
5. Runs the training loop, calling the attacker's methods.
6. Withholds the attack for the first few epochs.
7. Saves the final backdoored model weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from typing import Tuple

# --- 1. Add Project Root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Project-specific Imports ---
try:
    from config.cifar10 import *
    from data.cifar10_loader import load_cifar10, get_vsl_dataloaders
    from src.models.participant_vgg import ParticipantVGG16
    from src.models.server_model import ServerModel
    from src.attack.attacker import AttackerParticipant
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Details: {e}")
    print(f"Please ensure all modules (config, data, src.models, src.attack) are correctly defined.")
    sys.exit(1)

# --- 3. Attack Configuration ---
NUM_EPOCHS = 50  # Total training epochs
TARGET_LABEL = 3 # e.g., 'cat' in CIFAR-10
# Withhold swapping for first few epochs to stabilize
SWAP_DELAY_EPOCHS = 5  # 

MODEL_SAVE_DIR = os.path.join(project_root, "results", "models")
ATTACK_A_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_A.pth")
ATTACK_B_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_B.pth")
ATTACK_S_PATH = os.path.join(MODEL_SAVE_DIR, "backdoored_model_S.pth")


def find_seed_sample(dataset: torch.utils.data.Dataset, 
                     target_label: int, 
                     device: torch.device) -> Tuple[torch.Tensor, int]:
    """
    Finds the first sample in the dataset matching the target label
    to "seed" the attacker.
    """
    print(f"Searching for first sample with target label {target_label}...")
    for i in range(len(dataset)):
        # Assumes data/cifar10_loader.py has been modified
        # to have original_dataset return (image, label)
        try:
            image, label = dataset.dataset[dataset.indices[i]]
        except AttributeError:
             image, label = dataset[i] # Fallback for non-subset


        if label == target_label:
            print(f"Found seed sample at original index {i}.")
            # Add batch dimension: (C, H, W) -> (1, C, H, W)
            return image.unsqueeze(0).to(device), label
    
    raise RuntimeError(f"Could not find any sample with target label {target_label} in the dataset.")


def main():
    """
    Main function to orchestrate the VILLAIN attack.
    """
    
    # --- 1. Setup ---
    print("--- 🚀 Starting VILLAIN Attack Training ---")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is.available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Target Label: {TARGET_LABEL}")
    print(f"Attack starts after Epoch {SWAP_DELAY_EPOCHS}")

    # --- 2. Load Data ---
    print("\nLoading CIFAR-10 dataset...")
    trainset, testset = load_cifar10()
    
    # CRITICAL ASSUMPTION:
    # We assume 'data/cifar10_loader.py' is modified so that
    # its _SplitCIFAR10Dataset (for 'Server') returns (label, idx)
    # This is required for the EmbeddingSwapper's stateful dictionary.
    train_loader_A, train_loader_B, train_loader_S = get_vsl_dataloaders(
        original_dataset=trainset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=True
    )
    print("Data loaders created.")

    # --- 3. Find Attacker Seed Sample ---
    seed_data, seed_label = find_seed_sample(trainset, TARGET_LABEL, DEVICE)

    # --- 4. Initialize Models ---
    print("Initializing models...")
    model_A_malicious = ParticipantVGG16().to(DEVICE)
    model_B = ParticipantVGG16().to(DEVICE)
    model_S = ServerModel(
        participant_embedding_dim=model_A_malicious.embedding_dim,
        num_participants=NUM_PARTICIPANTS,
        num_classes=10
    ).to(DEVICE)

    # --- 5. Initialize Optimizers & Criterion ---
    # Attacker uses an increased learning rate [cite: 38]
    optimizer_A_malicious = optim.Adam(model_A_malicious.parameters(), lr=ATTACKER_LR)
    optimizer_B = optim.Adam(model_B.parameters(), lr=BENIGN_LR)
    optimizer_S = optim.Adam(model_S.parameters(), lr=SERVER_LR)
    
    criterion = nn.CrossEntropyLoss()
    print(f"Optimizers initialized (Attacker LR: {ATTACKER_LR}, Benign LR: {BENIGN_LR})")

    # --- 6. Initialize AttackerParticipant ---
    attacker = AttackerParticipant(
        model_a=model_A_malicious,
        optimizer_a=optimizer_A_malicious,
        target_label=TARGET_LABEL,
        n_candidates=N_CANDIDATES,
        theta=GRADIENT_RATIO_THRESHOLD,
        poisoning_rate=POISONING_RATE,
        trigger_magnitude=TRIGGER_MAGNITUDE_BETA,
        aug_dropout_ratio=AUG_DROPOUT_RATIO,
        aug_shifting_range=AUG_SHIFTING_RANGE,
        device=DEVICE,
        embed_dim=model_A_malicious.embedding_dim
    )
    print("AttackerParticipant object created.")

    # --- 7. Training Loop ---
    print("\n--- Starting Training Loop ---")
    for epoch in range(NUM_EPOCHS):
        
        # --- Check for Attack Activation ---
        if epoch == SWAP_DELAY_EPOCHS:
            print("--- !!! ---")
            print(f"Epoch {epoch+1}: Activating VILLAIN Label Inference.")
            attacker.seed_attacker(seed_data, seed_label)
            print("--- !!! ---")
        
        # Set models to train mode
        model_A_malicious.train()
        model_B.train()
        model_S.train()
        
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # ASSUMPTION: loader_S yields (labels, global_indices)
        batch_iterator = zip(loader_A, loader_B, loader_S)
        
        for data_A, data_B, (labels_S, global_indices) in batch_iterator:
            
            # Move data to device
            data_A = data_A.to(DEVICE)
            data_B = data_B.to(DEVICE)
            labels_S = labels_S.to(DEVICE)
            global_indices = global_indices.to(DEVICE) # Ensure indices are on device

            # --- 1. Zero all gradients ---
            optimizer_A_malicious.zero_grad()
            optimizer_B.zero_grad()
            optimizer_S.zero_grad()

            # --- 2. Step I: Participants' Forward Pass ---
            # Benign Participant
            e_B = model_B(data_B)
            # Malicious Participant
            e_to_upload, e_a_real = attacker.train_step(data_A, global_indices)

            # --- 3. Step II: Server's Forward & Backward Pass ---
            e_A_server = e_to_upload.detach().requires_grad_()
            e_B_server = e_B.detach().requires_grad_()

            e_agg = torch.cat((e_A_server, e_B_server), dim=1)
            
            logits = model_S(e_agg)
            loss = criterion(logits, labels_S)

            loss.backward()
            optimizer_S.step()

            # --- 4. Step III: Participants' Backward Pass ---
            # Server "sends back" gradients
            g_A = e_A_server.grad
            g_B = e_B_server.grad
            
            # Benign Participant update
            e_B.backward(g_B)
            optimizer_B.step()
            
            # Malicious Participant update (includes inference)
            attacker.backward_step(g_A, e_a_real, global_indices)

            # --- 5. Update Statistics ---
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels_S.size(0)
            correct_preds += (predicted == labels_S).sum().item()

        # --- 6. Log Epoch Statistics ---
        avg_epoch_loss = running_loss / len(loader_A)
        epoch_accuracy = 100.0 * correct_preds / total_samples
        
        attack_status = "INACTIVE" if epoch < SWAP_DELAY_EPOCHS else "ACTIVE"
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
              f"Loss: {avg_epoch_loss:.4f} | "
              f"Train Accuracy: {epoch_accuracy:.2f}% | "
              f"Attack: {attack_status}")

    # --- 8. Save Final Backdoored Models ---
    print("\n--- Training Complete ---")
    try:
        torch.save(model_A_malicious.state_dict(), ATTACK_A_PATH)
        torch.save(model_B.state_dict(), ATTACK_B_PATH)
        torch.save(model_S.state_dict(), ATTACK_S_PATH)
        print(f"Backdoored models saved to {MODEL_SAVE_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")
        
    print("--- ✅ VILLAIN Attack Script Finished ---")


if __name__ == "__main__":
    main()