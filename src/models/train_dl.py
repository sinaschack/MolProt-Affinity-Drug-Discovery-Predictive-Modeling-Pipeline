import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from ffnn import FFNN_Simple, FFNN_Deep, FFNN_Wide

# Load features
with open("data/pickle/egfr_combined.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X_combined"]
y = data["y"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training function (same as before)
def train_model(model, name, epochs=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = np.mean(val_losses)
        print(f"{name} | Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"models/{name}.pth")
        else:
            counter += 1
            if counter >= patience:
                print(f"{name}: Early stopping triggered")
                break

    print(f"{name} finished. Model saved.")

# Train all three
train_model(FFNN_Simple(X.shape[1]), "ffnn_egfr_simple")
train_model(FFNN_Deep(X.shape[1]), "ffnn_egfr_deep")
train_model(FFNN_Wide(X.shape[1]), "ffnn_egfr_wide")
