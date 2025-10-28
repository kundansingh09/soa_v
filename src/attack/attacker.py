"""
Defines the malicious participant's logic for the VILLAIN attack.

This module contains the AttackerParticipant class, which replaces
a benign participant in the VSL training. It manages both
Module 1 (Label Inference) and Module 2 (Data Poisoning).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Set, List, Optional, Tuple
import sys
import os

# --- Module Imports ---
# (Assumes project root is in sys.path)
try:
    from src.attack.label_inference import CandidateSelectionModel, EmbeddingSwapper
    from src.models.participant_model import ParticipantResNet18 # or ParticipantVGG16, or just nn.Module
        
    # These are assumed to exist, as per the paper's description
    from src.attack.data_poisoning import (
        get_trigger_mask, 
        fabricate_trigger, 
        augment_trigger
    )
except ImportError:
    # Add project root to path for standalone execution
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.attack.label_inference import CandidateSelectionModel, EmbeddingSwapper
    from src.models.participant_vgg import ParticipantVGG16
    
    # Stub functions if data_poisoning.py doesn't exist yet
    print("Warning: data_poisoning.py not found. Using stub functions.")
    def get_trigger_mask(*args, **kwargs): return torch.zeros(512)
    def fabricate_trigger(*args, **kwargs): return torch.zeros(512)
    def augment_trigger(*args, **kwargs): return torch.zeros(512)


class AttackerParticipant:
    """
    Implements the logic for the malicious participant (Attacker).
    
    This class wraps the attacker's local model and orchestrates
    the label inference and data poisoning modules.
    """
    def __init__(self,
                 model_a: nn.Module,
                 optimizer_a: optim.Optimizer,
                 target_label: int,
                 n_candidates: int,
                 theta: float,
                 poisoning_rate: float,
                 trigger_magnitude: float,
                 aug_dropout_ratio: float,
                 aug_shifting_range: List[float],
                 device: torch.device,
                 embed_dim: int = 512,
                 h_model_lr: float = 0.001):
        
        self.model_a = model_a
        self.optimizer_a = optimizer_a  # This optimizer has the *increased* LR
        self.device = device
        
        # Attack parameters
        self.target_label = target_label
        self.poisoning_rate = poisoning_rate
        
        # Module 1: Label Inference
        self.n_candidates = n_candidates
        self.candidate_model = CandidateSelectionModel(embed_dim).to(device)
        self.optimizer_h = optim.Adam(self.candidate_model.parameters(), lr=h_model_lr)
        self.criterion_h = nn.BCEWithLogitsLoss()
        self.embedding_swapper: Optional[EmbeddingSwapper] = None
        self.is_seeded = False
        self.inferred_target_indices: Set[int] = set()

        # Module 2: Data Poisoning
        self.trigger_fabrication_params = {
            'beta': trigger_magnitude
        }
        self.trigger_augmentation_params = {
            'dropout_ratio': aug_dropout_ratio,
            'shifting_range': aug_shifting_range
        }
        self.trigger_mask: Optional[torch.Tensor] = None
        self.trigger: Optional[torch.Tensor] = None
        self._all_embeddings_for_std: List[torch.Tensor] = []
        self._trigger_creation_threshold = 200 # Batches to collect for std dev


    def seed_attacker(self, seed_data: torch.Tensor, seed_label: int):
        """
        Initializes the EmbeddingSwapper with one known target sample.
        
        Args:
            seed_data: A single data sample (e.g., [1, C, H, W])
            seed_label: The label (must match target_label)
        """
        if seed_label != self.target_label:
            raise ValueError("Seed label does not match attacker's target label.")
        
        print(f"Attacker is seeding with one sample for target {self.target_label}...")
        self.model_a.eval()
        with torch.no_grad():
            seed_embedding = self.model_a(seed_data.to(self.device)).squeeze()
        self.model_a.train()
        
        self.embedding_swapper = EmbeddingSwapper(
            known_target_embedding=seed_embedding,
            theta=self.trigger_fabrication_params.get('theta', 1.5), # Config
            device=self.device
        )
        self.is_seeded = True
        print("Attacker seeding complete. Label inference is active.")

    def _update_candidate_model(self, newly_inferred_samples: List[Tuple]):
        """
        Trains the CandidateSelectionModel (H) in a semi-supervised
        manner using the results from embedding swapping.
        """
        if not newly_inferred_samples:
            return

        # Unpack the (global_idx, embedding, is_target) tuples
        embeddings = torch.stack([s[1] for s in newly_inferred_samples]).to(self.device)
        labels = torch.tensor([float(s[2]) for s in newly_inferred_samples]).unsqueeze(1).to(self.device)

        # Train H
        self.optimizer_h.zero_grad()
        logits = self.candidate_model(embeddings)
        loss = self.criterion_h(logits, labels)
        loss.backward()
        self.optimizer_h.step()

    def _get_or_create_trigger(self) -> Optional[torch.Tensor]:
        """
        Manages the trigger. If not created, it attempts to
        create it once enough embeddings are collected.
        """
        if self.trigger is not None:
            return self.trigger

        # Collect more embeddings to get a stable std dev
        if len(self._all_embeddings_for_std) < self._trigger_creation_threshold:
            return None

        print("Attacker: Creating trigger...")
        all_embeds = torch.cat(self._all_embeddings_for_std, dim=0)
        
        # Get mask (e.g., top 10% high-variance features)
        m = int(all_embeds.shape[1] * 0.1) 
        self.trigger_mask = get_trigger_mask(all_embeds, m=m)
        
        # Get trigger
        std_dev = all_embeds.std(dim=0)
        self.trigger = fabricate_trigger(
            self.trigger_mask,
            std_dev,
            self.trigger_fabrication_params['beta']
        ).to(self.device)
        
        # Clear buffer
        self._all_embeddings_for_std = [] 
        print("Attacker: Trigger created. Data poisoning is active.")
        return self.trigger

    def train_step(self,
                   batch_data_a: torch.Tensor,
                   batch_global_indices: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the attacker's forward pass.
        
        Returns:
            e_to_upload: The embeddings to send to the server.
            e_a_real: The *real* embeddings (kept for backward pass).
        """
        self.optimizer_a.zero_grad()
        
        # 1. Compute real embeddings
        e_a_real = self.model_a(batch_data_a)
        
        # Store for std dev calculation
        if self.trigger is None:
            self._all_embeddings_for_std.append(e_a_real.detach().cpu())
        
        e_to_upload = e_a_real.clone()

        # 2. Module 1: Label Inference
        if self.is_seeded:
            top_n_indices = self.candidate_model.predict_top_n(
                e_a_real.detach(), self.n_candidates
            )
            e_to_upload = self.embedding_swapper.get_embeddings_to_upload(
                e_a_real, batch_global_indices, top_n_indices
            )
        
        # 3. Module 2: Data Poisoning
        active_trigger = self._get_or_create_trigger()
        if active_trigger is not None:
            for i in range(len(batch_global_indices)):
                global_idx = batch_global_indices[i].item()
                
                # Poison only inferred target samples
                if global_idx in self.inferred_target_indices:
                    # And only at the specified poison rate
                    if random.random() < self.poisoning_rate:
                        
                        # Augment trigger for this sample
                        aug_trigger = augment_trigger(
                            active_trigger,
                            self.trigger_mask,
                            **self.trigger_augmentation_params
                        )
                        
                        # Apply additive trigger
                        e_to_upload[i] = e_a_real[i] + aug_trigger

        return e_to_upload, e_a_real

    def backward_step(self,
                      g_a: torch.Tensor,
                      e_a_real: torch.Tensor,
                      batch_global_indices: torch.Tensor):
        """
        Performs the attacker's backward pass.
        
        Args:
            g_a: The gradient from the server (g_k = \nabla_{e^k} L).
            e_a_real: The *original* embedding tensor from train_step.
            batch_global_indices: The global indices of the samples.
        """
        
        # 1. Module 1: Process Gradients for Label Inference
        if self.is_seeded and self.embedding_swapper:
            newly_inferred = self.embedding_swapper.process_gradients(
                g_a.detach(), 
                batch_global_indices, 
                e_a_real.detach()
            )
            
            # Update our set of known targets
            for idx, _, is_target in newly_inferred:
                if is_target:
                    self.inferred_target_indices.add(idx)
            
            # Train the candidate model H
            self._update_candidate_model(newly_inferred)

        # 2. Module 2: Update Attacker's Model
        # Backpropagate the server's gradient through f^a
        e_a_real.backward(g_a)
        
        # Update f^a using the *increased* learning rate
        self.optimizer_a.step()