import torch
from airwriting_imu.core.ctc_model import CTCRecognizer

def test_ctc():
    B = 4
    max_T = 100
    input_dim = 11
    
    # Dummy data
    x = torch.randn(B, max_T, input_dim)
    lengths = torch.tensor([100, 80, 50, 10], dtype=torch.long)
    
    targets = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2])
    target_lengths = torch.tensor([2, 3, 2, 1], dtype=torch.long)
    
    model = CTCRecognizer(input_dim=input_dim, num_classes=26)
    model.train()
    
    print("Testing forward pass with pack_padded_sequence...")
    try:
        log_probs = model(x, lengths=lengths)
        print("Forward pass successful. Shape:", log_probs.shape)
    except Exception as e:
        print("Forward pass failed:", e)
        return
        
    print("Testing CTC Loss computation...")
    try:
        loss = model.compute_loss(log_probs, targets, lengths, target_lengths)
        print("Loss computed successfully. Loss:", loss.item())
    except Exception as e:
        print("Loss computation failed:", e)
        return
        
    print("Testing backward pass...")
    try:
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print("Backward pass failed:", e)
        return
        
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test_ctc()
