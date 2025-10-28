"""
Implements Module 1 of the VILLAIN attack: Label Inference.

Contains the Candidate Selection Model (H) and the EmbeddingSwapper
which uses gradient side-channels to infer labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict, Set

# --- 1. Candidate Selection Model (H) ---

class CandidateSelectionModel(nn.Module):
    """
    [cite_start]Implements the binary classifier H described in[cite: 251].
    
    This model predicts whether a given embedding belongs to the
    target class. It is a simple Multi-Layer Perceptron (MLP).
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super(CandidateSelectionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.embedding_dim = embedding_dim

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw logit (pre-sigmoid) for binary classification.
        A higher value means more likely to be the target class.
        """
        return self.layers(embedding)

    def predict_top_n(self,
                      batch_embeddings: torch.Tensor,
                      n: int) -> torch.Tensor:
        """
        Selects the top-n candidate samples from a batch that are
        [cite_start]most likely to belong to the target label[cite: 257].

        Args:
            batch_embeddings: The batch of embeddings (B, E).
            n: The number of candidates to select (N_CANDIDATES).

        Returns:
            A 1D tensor of the batch *indices* of the top-n candidates.
        """
        with torch.no_grad():
            logits = self.forward(batch_embeddings).squeeze()
            
            # Handle the case where batch size is less than n
            k = min(n, len(logits))
            
            # Get the indices of the top-k highest logits
            _, top_n_indices = torch.topk(logits, k=k)
            
        return top_n_indices

# --- 2. Embedding Swapping Logic ---

class EmbeddingSwapper:
    """
    [cite_start]Manages the stateful "Embedding Swapping" attack[cite: 226].
    
    This class tracks the state of each sample (e.g., "waiting for
    baseline gradient", "waiting for swapped gradient") and
    performs the label inference based on the 2-step gradient
    [cite_start]comparison [cite: 234-237].
    """
    def __init__(self,
                 known_target_embedding: torch.Tensor,
                 theta: float,
                 device: torch.device):
        """
        Args:
            [cite_start]known_target_embedding: The 'seed' embedding e_t^a[cite: 230].
            [cite_start]theta: The gradient ratio threshold θ[cite: 234].
            device: The device for tensor operations.
        """
        # Group of embeddings known/inferred to be target class
        # [cite_start]Used for "Inference Adjustment" [cite: 261-263]
        self.target_label_embedding_group: List[torch.Tensor] = \
            [known_target_embedding.detach().to(device)]
        
        # Stores {sample_global_idx: baseline_gradient_norm}
        self.baseline_gradients: Dict[int, float] = {}
        
        # Tracks the state of samples currently being tested
        # {sample_global_idx: 'wait_for_baseline' or 'wait_for_swap'}
        self.pending_swaps: Dict[int, str] = {}
        
        self.theta = theta  # Gradient ratio threshold θ
        self.mu = float('inf')  # Gradient norm threshold μ, updated externally
        self.device = device

    def update_mu(self, avg_grad_norm: float):
        """
        [cite_start]Updates the dynamic gradient norm threshold μ[cite: 334].
        """
        self.mu = avg_grad_norm

    def _get_random_target_embedding(self) -> torch.Tensor:
        """
        Implements "Inference Adjustment" by randomly selecting
        [cite_start]an embedding from the group to be stealthy[cite: 263].
        """
        return random.choice(self.target_label_embedding_group)

    def get_embeddings_to_upload(
        self,
        real_batch_embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        top_n_candidate_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Determines which embeddings to send to the server.
        Sends the real embedding for baseline, or a swapped
        embedding for testing.

        Args:
            real_batch_embeddings: The true embeddings for this batch.
            batch_indices: The global indices of the samples in this batch.
            top_n_candidate_indices: The batch indices of the top-n candidates.

        Returns:
            A tensor of embeddings to be uploaded to the server.
        """
        embeddings_to_upload = real_batch_embeddings.clone()
        
        for batch_idx in top_n_candidate_indices:
            global_idx = batch_indices[batch_idx].item()

            if global_idx not in self.baseline_gradients:
                # --- Step A: Get Baseline Gradient ---
                # This is our first time testing this sample.
                # We upload the *real* embedding to get g_i^a.
                # We mark it as pending.
                self.pending_swaps[global_idx] = 'wait_for_baseline'
                # (No swap is needed, embeddings_to_upload[batch_idx]
                # is already the real one)
                
            elif global_idx not in self.pending_swaps:
                # --- Step B: Get Swapped Gradient ---
                # We have a baseline, now we test.
                # We upload a *swapped* embedding to get g_hat_i^a.
                swapped_embedding = self._get_random_target_embedding()
                embeddings_to_upload[batch_idx] = swapped_embedding
                self.pending_swaps[global_idx] = 'wait_for_swap'

        return embeddings_to_upload

    def process_gradients(
        self,
        batch_gradients: torch.Tensor,
        batch_indices: torch.Tensor,
        real_batch_embeddings: torch.Tensor
    ) -> List[Tuple[int, torch.Tensor, bool]]:
        """
        Processes the received gradients to perform inference.
        This is the core of the 2-step comparison.

        Args:
            batch_gradients: The gradients g^k received from the server.
            batch_indices: The global indices of the samples.
            real_batch_embeddings: The *original* embeddings for the batch.

        Returns:
            A list of (global_idx, real_embedding, is_target_label)
            tuples for newly inferred samples. This is used to
            train the CandidateSelectionModel.
        """
        newly_inferred_samples = []

        for i, global_idx in enumerate(batch_indices):
            global_idx = global_idx.item()
            state = self.pending_swaps.get(global_idx)
            
            if state is None:
                continue  # This sample wasn't a candidate

            grad_norm = batch_gradients[i].norm().item()
            real_embedding = real_batch_embeddings[i].detach()

            if state == 'wait_for_baseline':
                # --- Step A Complete ---
                # [cite_start]This is the baseline gradient g_i^a[cite: 231].
                self.baseline_gradients[global_idx] = grad_norm
                del self.pending_swaps[global_idx]
            
            elif state == 'wait_for_swap':
                # --- Step B Complete ---
                # [cite_start]This is the swapped gradient g_hat_i^a[cite: 232].
                baseline_norm = self.baseline_gradients.get(global_idx)
                
                # Failsafe, should always have a baseline
                if baseline_norm is None:
                    del self.pending_swaps[global_idx]
                    continue
                
                # Avoid division by zero
                if baseline_norm < 1e-9:
                    ratio = float('inf')
                else:
                    ratio = grad_norm / baseline_norm
                
                # [cite_start]--- The Inference Logic [cite: 234-237] ---
                is_target = (baseline_norm <= self.mu) and (ratio <= self.theta)
                
                if is_target:
                    # SUCCESS: Inferred a new target sample.
                    # Add its *real* embedding to the group for
                    # [cite_start]"Inference Adjustment"[cite: 262].
                    self.target_label_embedding_group.append(real_embedding)
                
                # [cite_start]Add to list to train classifier H [cite: 256]
                newly_inferred_samples.append((global_idx, real_embedding, is_target))
                del self.pending_swaps[global_idx]

        return newly_inferred_samples


# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing Label Inference Module ---")

    # 1. Load config
    try:
        from config.cifar10 import (
            BATCH_SIZE, N_CANDIDATES, 
            GRADIENT_RATIO_THRESHOLD
        )
        EMBED_DIM = 512 # From participant_vgg.py
    except ImportError:
        print("Could not load config, using defaults.")
        BATCH_SIZE = 4
        N_CANDIDATES = 2
        GRADIENT_RATIO_THRESHOLD = 1.5
        EMBED_DIM = 512

    DEVICE = torch.device("cpu")

    # 2. Init components
    print(f"Initializing CandidateSelectionModel (H) with embed_dim={EMBED_DIM}...")
    candidate_model = CandidateSelectionModel(EMBED_DIM).to(DEVICE)

    known_embed = torch.randn(EMBED_DIM, device=DEVICE)
    swapper = EmbeddingSwapper(known_embed, GRADIENT_RATIO_THRESHOLD, DEVICE)
    
    # Set a mock dynamic threshold μ
    swapper.update_mu(avg_grad_norm=10.0)
    print(f"Swapper initialized. mu=10.0, theta={swapper.theta}")

    # 3. --- Simulate Batch 1 (Get Baselines) ---
    print("\n--- Simulating Batch 1 (Indices 0-3) ---")
    b1_embeds = torch.randn(BATCH_SIZE, EMBED_DIM, device=DEVICE)
    b1_indices = torch.arange(BATCH_SIZE, device=DEVICE)
    
    # H selects candidates (e.g., indices 1, 3 in the batch)
    top_n_indices = candidate_model.predict_top_n(b1_embeds, N_CANDIDATES)
    print(f"H selected batch indices: {top_n_indices.tolist()}")

    embeds_to_upload = swapper.get_embeddings_to_upload(
        b1_embeds, b1_indices, top_n_indices
    )
    print(f"Pending swaps: {swapper.pending_swaps}")
    
    # Server sends back gradients (all are baseline grads)
    # Let's make grad for idx=1 "good" (low norm)
    # and grad for idx=3 "bad" (high norm)
    b1_grads = torch.randn(BATCH_SIZE, EMBED_DIM, device=DEVICE)
    b1_grads[top_n_indices[0]] *= 1.0  # Mock norm ~1.0
    b1_grads[top_n_indices[1]] *= 15.0 # Mock norm ~15.0
    
    inferred = swapper.process_gradients(b1_grads, b1_indices, b1_embeds)
    
    print(f"Baseline grads stored: {swapper.baseline_gradients}")
    print(f"Inferred samples: {inferred}")
    print(f"Pending swaps: {swapper.pending_swaps}")

    # 4. --- Simulate Batch 2 (Perform Swaps) ---
    print("\n--- Simulating Batch 2 (Same Indices 0-3) ---")
    # New embeddings for same global indices
    b2_embeds = torch.randn(BATCH_SIZE, EMBED_DIM, device=DEVICE)
    b2_indices = torch.arange(BATCH_SIZE, device=DEVICE)

    top_n_indices = candidate_model.predict_top_n(b2_embeds, N_CANDIDATES)
    print(f"H selected batch indices: {top_n_indices.tolist()}")
    
    embeds_to_upload = swapper.get_embeddings_to_upload(
        b2_embeds, b2_indices, top_n_indices
    )
    print(f"Pending swaps: {swapper.pending_swaps}")
    # Check if swap happened (tensor values will be different)
    assert not torch.allclose(embeds_to_upload[top_n_indices[0]], b2_embeds[top_n_indices[0]])
    print("Swap confirmed: Embeddings to upload differ from real embeddings.")
    
    # Server sends back gradients (swapped grads)
    # For idx=1 (good baseline): make swapped grad also good (low ratio)
    # For idx=3 (bad baseline): make swapped grad also bad (low ratio)
    b2_grads = torch.randn(BATCH_SIZE, EMBED_DIM, device=DEVICE)
    b2_grads[top_n_indices[0]] *= 1.2  # norm ~1.2. Ratio ~1.2/1.0 = 1.2 (<= theta)
    b2_grads[top_n_indices[1]] *= 18.0 # norm ~18.0. Ratio ~18.0/15.0 = 1.2 (<= theta)

    inferred = swapper.process_gradients(b2_grads, b2_indices, b2_embeds)

    print(f"Inferred samples: {inferred}")
    print(f"Pending swaps: {swapper.pending_swaps}")
    print(f"Target embedding group size: {len(swapper.target_label_embedding_group)}")
    
    # 5. Verification
    # idx=1: baseline=1.0 (<= mu), ratio=1.2 (<= theta) -> TRUE
    # idx=3: baseline=15.0 (> mu), ratio=1.2 (<= theta) -> FALSE
    assert len(inferred) == 2
    assert inferred[0][2] == True  # idx=1 is target
    assert inferred[1][2] == False # idx=3 is not target
    assert len(swapper.target_label_embedding_group) == 2 # Initial + inferred
    
    print("\nSUCCESS: Label inference logic is correct.")