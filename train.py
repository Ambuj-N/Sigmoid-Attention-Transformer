import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import SigmoidTransformerBlock
from baseline import SoftmaxTransformerBlock

def get_batch(batch_size=32, seq_length=16, d_model=8):
    data = torch.randn(batch_size, seq_length, d_model)
    return data, data

def main():
    print("Initializing models...")
    d_model = 8
    learning_rate = 0.0001        
    iterations = 300

    sigmoid_model = SigmoidTransformerBlock(d_model)
    softmax_model = SoftmaxTransformerBlock(d_model)

    criterion = nn.MSELoss()
    opt_sigmoid = torch.optim.Adam(sigmoid_model.parameters(), lr=learning_rate)
    opt_softmax = torch.optim.Adam(softmax_model.parameters(), lr=learning_rate)

    sig_losses = []
    soft_losses = []

    print("Starting the head-to-head training race...")
    for i in range(iterations):
        x, y = get_batch(batch_size=32, seq_length=16, d_model=d_model)

        opt_sigmoid.zero_grad()
        out_sig = sigmoid_model(x)
        loss_sig = criterion(out_sig, y)
        loss_sig.backward()
        opt_sigmoid.step()
        sig_losses.append(loss_sig.item())

        opt_softmax.zero_grad()
        out_soft = softmax_model(x)
        loss_soft = criterion(out_soft, y)
        loss_soft.backward()
        opt_softmax.step()
        soft_losses.append(loss_soft.item())

        if (i + 1) % 5 == 0:
            print(f"Step {i+1:<4} | Sigmoid Loss: {loss_sig.item():.4f} | Softmax Loss: {loss_soft.item():.4f}")

    print("Training complete! Generating performance graph...")
    
    plt.figure(figsize=(10, 5))
    plt.plot(sig_losses, label="Sigmoid Attention (Stable)", color="blue", alpha=0.8)
    plt.plot(soft_losses, label="Softmax Attention (Baseline)", color="red", alpha=0.8)
    plt.title("Training Loss: Sigmoid vs. Softmax Attention")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Squared Error Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("loss_comparison.png")
    print("Graph saved successfully as 'loss_comparison.png'!")

if __name__ == "__main__":
    main()
