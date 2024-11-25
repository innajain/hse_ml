import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os
import matplotlib.pyplot as plt


def load_data(input_file, output_file):
    inputs = pd.read_csv(input_file, header=None).values
    outputs = pd.read_csv(output_file, header=None).values
    inputs = np.nan_to_num(inputs, nan=0)
    # outputs = np.repeat(outputs, 10, axis=0)
    return inputs, outputs

def preprocess_data(inputs, outputs, power_scaler=None):
    # input scaling
    power_inputs = inputs[:, :108]
    voltage_mag = inputs[:, 108:108+14]
    voltage_angle = inputs[:, 108+14: 108+14+14]
    from_current_mag = inputs[:, 108+14+14: 108+14+14+20]
    from_current_angle = inputs[:, 108+14+14+20: 108+14+14+20+20]
    to_current_mag = inputs[:, 108+14+14+20+20: 108+14+14+20+20+20]
    to_current_angle = inputs[:, 108+14+14+20+20+20: 108+14+14+20+20+20+20]

    if power_scaler is None:
        power_scaler = RobustScaler()
        power_scaler.fit(power_inputs)

    power_inputs = power_scaler.transform(power_inputs)

    voltage_angle_rad = np.radians(voltage_angle)
    voltage_angle_sin = np.sin(voltage_angle_rad)
    voltage_angle_cos = np.cos(voltage_angle_rad)

    from_current_angle_rad = np.radians(from_current_angle)
    from_current_angle_sin = np.sin(from_current_angle_rad)
    from_current_angle_cos = np.cos(from_current_angle_rad)

    to_current_angle_rad = np.radians(to_current_angle)
    to_current_angle_sin = np.sin(to_current_angle_rad)
    to_current_angle_cos = np.cos(to_current_angle_rad)
    
    inputs_processed = np.hstack([power_inputs, voltage_mag, voltage_angle_sin, voltage_angle_cos, from_current_mag, from_current_angle_sin, from_current_angle_cos, to_current_mag, to_current_angle_sin, to_current_angle_cos])

    # output scaling
    voltage_mag_output = outputs[:, :14]
    voltage_angle_output = outputs[:, 14:]

    voltage_angle_output_rad = np.radians(voltage_angle_output[:, 1:])
    voltage_angle_output_sin = np.sin(voltage_angle_output_rad)
    voltage_angle_output_cos = np.cos(voltage_angle_output_rad)

    outputs_processed = np.hstack([voltage_mag_output, voltage_angle_output_sin, voltage_angle_output_cos])
    
    return inputs_processed, outputs_processed, power_scaler


class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        scale = torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32, device=x.device))
        weights = self.attention_weights(x) / scale

        if mask is not None:
            # Replace weights for masked features with a large negative value to ignore them in softmax
            weights = weights.masked_fill(mask == 0, float('-inf'))

        weights = self.softmax(weights)
        return weights * x

class StateEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StateEstimator, self).__init__()
        self.attention = AttentionBlock(input_dim)
        hidden_dim1 = 256
        hidden_dim2 = 128
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim1)
        self.bn3 = nn.BatchNorm1d(hidden_dim2)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_head = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output_head.weight)

    def forward(self, x):
        x = self.attention(x)
        x = self.bn1(x)
        x = torch.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout(x)
        return self.output_head(x)

class CombinedLoss(nn.Module):
    def __init__(self, voltage_weight=1.0, angle_weight=1.0, l1_weight=0.1, epsilon=1e-10):
        super().__init__()
        self.voltage_weight = voltage_weight
        self.angle_weight = angle_weight
        self.l1_weight = l1_weight
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Voltage Magnitude Loss: Normalized Percentage Error
        voltage_error = torch.abs(pred[:, :14] - target[:, :14]) / (torch.abs(target[:, :14]) + self.epsilon)
        voltage_loss = voltage_error.mean()

        # Phase Angle Loss: MSE in sine-cosine space
        angle_loss = nn.functional.mse_loss(pred[:, 14:], target[:, 14:])

        # L1 Regularization for smoothness
        l1_loss = nn.functional.l1_loss(pred, target)

        # Weighted combination of the losses
        total_loss = (self.voltage_weight * voltage_loss +
                      self.angle_weight * angle_loss +
                      self.l1_weight * l1_loss) / (self.voltage_weight + self.angle_weight + self.l1_weight)
        return total_loss

def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, outputs in train_loader:
            inputs, outputs = inputs.to(device), outputs.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, outputs)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}")
                continue
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.6f}")
        
        if avg_epoch_loss > 1e6:
            print("Loss too high, stopping training")
            break
            
    return losses

def undo_preprocessing_outputs(outputs):
    voltage_mags = outputs[:, :14]
    voltage_angles_sin = outputs[:, 14:14+13]
    voltage_angles_cos = outputs[:, 14+13:]
    voltage_angles = np.degrees(np.arctan2(voltage_angles_sin, voltage_angles_cos))
    voltage_angles = np.insert(voltage_angles, 0, 0, axis=1)
    return np.hstack([voltage_mags, voltage_angles])

def evaluate(model, test_loader, criterion, device):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)


    predictions = undo_preprocessing_outputs(predictions)
    targets = undo_preprocessing_outputs(targets)

    epsilon = 1e-10
    percentage_errors = np.abs((predictions - targets) / (np.abs(targets) + epsilon)) * 100
    avg_percentage_error = np.mean(percentage_errors, axis=0)
    max_percentage_error = np.max(percentage_errors, axis=0)

    return {
        "avg_loss": avg_loss,
        "avg_percentage_error": avg_percentage_error,
        "max_percentage_error": max_percentage_error,
        "predictions": predictions,
        "targets": targets
    }

def save_predictions_vs_targets_plot(predictions, targets, save_dir="plots", suffix=""):
    if len(predictions) != 28 or len(targets) != 28:
        raise ValueError("Both predictions and targets must have 28 elements.")

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Split into two groups for the first 14 and the rest
    predicted_first_14 = predictions[:14]
    target_first_14 = targets[:14]

    predicted_second_14 = predictions[14:]
    target_second_14 = targets[14:]

    # Define bar width and positions for grouped bars
    bar_width = 0.35
    indices = np.arange(14)

    # Plot and save the first 14 outputs
    plt.figure(figsize=(14, 6))
    plt.bar(indices - bar_width / 2, target_first_14, bar_width, label="Targets", color="blue")
    plt.bar(indices + bar_width / 2, predicted_first_14, bar_width, label="Predicted", color="red")
    plt.title("Voltage Magnitudes")
    plt.xlabel("Bus Number")
    plt.ylabel("Value (in p.u.)")
    plt.xticks(indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_path_1 = os.path.join(save_dir, f"magnitudes{suffix}.png")
    plt.savefig(save_path_1)
    plt.close()

    # Plot and save the second 14 outputs
    plt.figure(figsize=(14, 6))
    plt.bar(indices - bar_width / 2, target_second_14, bar_width, label="Targets", color="blue")
    plt.bar(indices + bar_width / 2, predicted_second_14, bar_width, label="Predicted", color="red")
    plt.title("Voltage Phase Angles")
    plt.xlabel("Bus Number")
    plt.ylabel("Value (in degrees)")
    plt.xticks(indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_path_2 = os.path.join(save_dir, f"angles{suffix}.png")
    plt.savefig(save_path_2)
    plt.close()

def save_loss_log_scale_plot(losses, save_path="plots/loss_log_plot.png"):
    """
    Saves the training loss vs. epochs plot on a logarithmic scale.

    Args:
        losses (list): List of average losses for each epoch.
        save_path (str): Path to save the plot image. Default is "loss_log_plot.png".
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', label='Training Loss')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title("Training Loss vs. Epochs (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(save_path)
    plt.close()  # Close the plot to avoid memory issues

def plot_error_distribution(predictions, targets, save_path="plots/error_distribution.png"):
    """
    Plots the error distribution for each output field.

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Target values.
        save_path (str): Path to save the plot image. Default is "error_distribution.png".
    """
    epsilon = 1e-10
    percentage_errors = np.abs((predictions - targets) / (np.abs(targets) + epsilon)) * 100

    plt.figure(figsize=(14, 6))
    # for i in range(percentage_errors.shape[1]):
    plt.plot(percentage_errors[:, 15], label=f'Output {15+1}')
    
    plt.title("Error Distribution for Each Output Field")
    plt.xlabel("Number of Test Cases")
    plt.ylabel("Percent Error")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    INPUT_FILE = "./inputs.csv"   
    OUTPUT_FILE = "./outputs.csv"
    MODEL_SAVE_PATH = "state_estimator_model.pth"
    useCheckpoint = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    inputs, outputs = load_data(INPUT_FILE, OUTPUT_FILE)
    print("Data loaded. Input shape:", inputs.shape, "Output shape:", outputs.shape)
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.2, random_state=45)
    inputs_train, outputs_train, power_scaler = preprocess_data(inputs_train, outputs_train)
    
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(inputs_train, dtype=torch.float32),
        torch.tensor(outputs_train, dtype=torch.float32),
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    input_dim = inputs_train.shape[1]
    output_dim = outputs_train.shape[1]
    model = StateEstimator(input_dim, output_dim).to(device)

    criterion = CombinedLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    if useCheckpoint and os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model checkpoint and optimizer state loaded.")
    else:
        losses = train_model(model, train_loader, criterion, optimizer, device, epochs=500)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, MODEL_SAVE_PATH)
        print("Model training complete.")


    inputs_test,outputs_test, power_scaler = preprocess_data(inputs_test, outputs_test, power_scaler)

    test_data = torch.utils.data.TensorDataset(
        torch.tensor(inputs_test, dtype=torch.float32),
        torch.tensor(outputs_test, dtype=torch.float32),
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    
    evaluation_results = evaluate(model, test_loader, criterion, device)

    print(f"Average Loss: {evaluation_results['avg_loss']:.6f}")
    save_predictions_vs_targets_plot(evaluation_results['predictions'][0], evaluation_results['targets'][0])
