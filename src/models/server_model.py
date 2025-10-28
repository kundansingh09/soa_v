"""
Defines the server's model for Vertical Split Learning.

As per the paper, this is a 3-layer fully-connected
[cite_start]neural network[cite: 319]. It takes the aggregated (concatenated)
embeddings from all participants as input and predicts the
final class label.
"""

import torch
import torch.nn as nn

class ServerModel(nn.Module):
    """
    Implements the 3-layer FC network for the server.
    """
    def __init__(self, 
                 participant_embedding_dim: int = 512, 
                 num_participants: int = 2, 
                 num_classes: int = 10):
        """
        Args:
            participant_embedding_dim: The output dimension
                of a single participant's model (e.g., 512 for VGG16).
            num_participants: The number of participants (e.g., 2).
            num_classes: The number of output classes (e.g., 10 for CIFAR-10).
        """
        super(ServerModel, self).__init__()
        
        # 1. Define the combined input dimension
        # [cite_start]Per the paper, the default aggregation is concatenation [cite: 331]
        self.input_dim = participant_embedding_dim * num_participants
        
        # 2. Define the hidden dimensions
        # These are not specified in the paper, so we assume
        # reasonable values for a 3-layer FC network.
        # H1 = 256, H2 = 128
        h1_dim = 256
        h2_dim = 128
        
        # 3. Define the 3-layer FC network
        self.fc_layers = nn.Sequential(
            # Layer 1
            nn.Linear(self.input_dim, h1_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # Add dropout for regularization
            
            # Layer 2
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            # Layer 3 (Output Layer)
            nn.Linear(h2_dim, num_classes)
        )

    def forward(self, aggregated_embedding: torch.Tensor) -> torch.Tensor:
        """
        Passes the aggregated embedding through the FC layers.
        
        Args:
            aggregated_embedding: A batch of concatenated embeddings
                from all participants, shape (B, 1024).
        
        Returns:
            The raw logits for classification, shape (B, 10).
        """
        logits = self.fc_layers(aggregated_embedding)
        return logits

# --- Test Block ---
if __name__ == '__main__':
    # 1. Load config
    try:
        from config.cifar10 import BATCH_SIZE
    except ImportError:
        print("Could not load config, using defaults.")
        BATCH_SIZE = 4
    
    # 2. Define model params
    P_EMBED_DIM = 512
    NUM_P = 2
    NUM_C = 10
    
    # 3. Define dummy input
    # Shape = (Batch, P_EMBED_DIM * NUM_P)
    dummy_input_shape = (BATCH_SIZE, P_EMBED_DIM * NUM_P)
    dummy_input = torch.randn(dummy_input_shape)
    
    print(f"--- Testing ServerModel ---")
    print(f"Dummy aggregated input shape: {dummy_input.shape}")

    # 4. Initialize model
    model = ServerModel(
        participant_embedding_dim=P_EMBED_DIM,
        num_participants=NUM_P,
        num_classes=NUM_C
    )
    print(f"Model Input Dim: {model.input_dim}")
    print(model)

    # 5. Test forward pass
    output_logits = model(dummy_input)
    print(f"\nOutput logits shape: {output_logits.shape}")

    # 6. Verification
    expected_shape = (BATCH_SIZE, NUM_C)
    assert output_logits.shape == expected_shape
    print(f"\nSUCCESS: Output shape {output_logits.shape} matches expected {expected_shape}.")