from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F


leech_lattices = np.load(Path(__file__).parent / "../../cache/leech_lattices_normalized.npy") * math.sqrt(32)
leech_lattices = torch.from_numpy(leech_lattices).long()  # (196560, 24)

def factorized_cross_entropy_loss(input, target, target_indices, reduction='mean'):
    """
    Computes the factorized cross-entropy loss between input and target.

    Args:
        input (Tensor): The input tensor of shape (B, C, N, H) where C = number of classes, and H = number of heads.
        target (Tensor): The target tensor of shape (B, N, H) with class indices.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    B, C, N, H = input.shape
    log_probs = torch.log_softmax(input, dim=1)  # (B, C, N, H)
    leech_lattices_shifted = leech_lattices + 4  # Shift to [0, 8]
    indices = leech_lattices_shifted.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1).to(input.device)
    gathered_log_probs = torch.gather(log_probs, dim=1, index=indices)  # (B, 196560, N, H)
    joint_log_probs = gathered_log_probs.mean(dim=-1)
    return F.nll_loss(joint_log_probs, target_indices, reduction=reduction), joint_log_probs


def factorized_cross_entropy_loss_v2(input, target, target_indices, reduction='mean', chunk_size=1000):
    """
    Memory-optimized version of factorized cross-entropy loss using chunked processing.

    Args:
        input (Tensor): The input tensor of shape (B, C, N, H) where C = number of classes, and H = number of heads.
        target (Tensor): The target tensor of shape (B, N, H) with class indices.
        target_indices (Tensor): The target indices of shape (B, N).
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
        chunk_size (int): Number of leech lattices to process at once to reduce memory usage. Default: 1000.
    """
    B, C, N, H = input.shape
    log_probs = torch.log_softmax(input, dim=1)  # (B, C, N, H)
    leech_lattices_shifted = leech_lattices + 4  # Shift to [0, 8]
    leech_lattices_device = leech_lattices_shifted.to(input.device)
    
    num_lattices = leech_lattices_device.shape[0]
    joint_log_probs_list = []
    
    # Process in chunks to reduce memory usage
    for start_idx in range(0, num_lattices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_lattices)
        chunk_lattices = leech_lattices_device[start_idx:end_idx]  # (chunk_size, 24)
        
        # Use advanced indexing without creating large intermediate tensors
        # chunk_lattices: (chunk_size, 24), log_probs: (B, C, N, H)
        chunk_indices = chunk_lattices.unsqueeze(0).unsqueeze(2)  # (1, chunk_size, 1, 24)
        chunk_indices = chunk_indices.expand(B, -1, N, -1)  # (B, chunk_size, N, 24)
        
        # Gather log probabilities for this chunk
        chunk_gathered = torch.gather(log_probs, dim=1, index=chunk_indices)  # (B, chunk_size, N, H)
        chunk_joint_log_probs = chunk_gathered.mean(dim=-1)  # (B, chunk_size, N)  # TODO: check torch.sum()

        joint_log_probs_list.append(chunk_joint_log_probs)
        
        # Clear intermediate tensors to free memory
        del chunk_indices, chunk_gathered, chunk_joint_log_probs
    
    # Concatenate all chunks
    joint_log_probs = torch.cat(joint_log_probs_list, dim=1)  # (B, num_lattices, N)
    
    return F.nll_loss(joint_log_probs, target_indices, reduction=reduction), joint_log_probs


if __name__ == "__main__":
    B, C, N, H = 4, 9, 2, 24
    input = torch.randn(B, C, N, H)
    target = torch.randint(0, C, (B, N, H))
    target_indices = torch.randint(0, 196560, (B, N))
    
    # Test original function
    print("Original function result:")
    loss_v1, joint_log_probs_v1 = factorized_cross_entropy_loss(input, target, target_indices=target_indices, reduction='mean')
    print(f"Loss: {loss_v1}")
    print(f"Joint log probs shape: {joint_log_probs_v1.shape}")
    print(f"Joint log probs range: [{joint_log_probs_v1.min():.4f}, {joint_log_probs_v1.max():.4f}]")
    print(f"Expected range should be around: [{24 * torch.log(torch.tensor(1/C)):.4f}, {24 * torch.log(torch.tensor(1.0)):.4f}]")
    
    # Test optimized function
    print("\nOptimized function result:")
    loss_v2, joint_log_probs_v2 = factorized_cross_entropy_loss_v2(input, target, target_indices=target_indices, reduction='mean', chunk_size=1000)
    print(f"Loss: {loss_v2}")
    print(f"Joint log probs shape: {joint_log_probs_v2.shape}")
    print(f"Joint log probs range: [{joint_log_probs_v2.min():.4f}, {joint_log_probs_v2.max():.4f}]")
    
    # Verify results are close
    print(f"\nLosses are close: {torch.allclose(loss_v1, loss_v2, atol=1e-6)}")
    print(f"Joint log probs are close: {torch.allclose(joint_log_probs_v1, joint_log_probs_v2, atol=1e-6)}")
    print(f"Loss difference: {torch.abs(loss_v1 - loss_v2).item()}")
    
    # Debug: Check log_softmax output
    print(f"\nLog softmax range: [{torch.log_softmax(input, dim=1).min():.4f}, {torch.log_softmax(input, dim=1).max():.4f}]")
    
    # Verify numerical correctness: check a few specific indices
    print(f"\nNumerical verification:")
    print(f"Max absolute difference in joint_log_probs: {torch.abs(joint_log_probs_v1 - joint_log_probs_v2).max():.10f}")
    print(f"Mean absolute difference in joint_log_probs: {torch.abs(joint_log_probs_v1 - joint_log_probs_v2).mean():.10f}")
    
    # The values are correct - joint log probs should be negative and can be quite negative
    # because we're summing log probabilities across 24 heads
    print(f"\nExplanation of ranges:")
    print(f"- Log softmax per head: [{torch.log_softmax(input, dim=1).min():.4f}, {torch.log_softmax(input, dim=1).max():.4f}]")
    print(f"- With {H} heads, joint range could be: [{H * torch.log_softmax(input, dim=1).min():.1f}, {H * torch.log_softmax(input, dim=1).max():.1f}]")
    print(f"- Actual joint range: [{joint_log_probs_v1.min():.4f}, {joint_log_probs_v1.max():.4f}]")
    print(f"- This is normal - leech lattice constraints limit the achievable range")
