import sys
import torch
import torch.nn as nn
from airwriting_imu.core.ctc_model import CTCRecognizer

def run():
    print("Test started")
    model = CTCRecognizer(num_classes=26, input_dim=11, hidden_dim=32, num_lstm_layers=1)
    
    x = torch.randn(2, 50, 11)
    lengths = torch.tensor([50, 20], dtype=torch.long)
    
    targets = torch.tensor([1, 2, 3])
    target_lengths = torch.tensor([2, 1], dtype=torch.long)
    
    try:
        log_probs = model(x, lengths=lengths)
        print("Forward passed. shape:", log_probs.shape)
        loss = model.compute_loss(log_probs, targets, lengths, target_lengths)
        print("Loss passed. loss:", loss.item())
        loss.backward()
        print("Backward passed.")
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    run()
