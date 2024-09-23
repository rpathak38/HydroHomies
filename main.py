import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the FloodDataset class
class FloodDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        super(FloodDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms

        self.flood_path = os.path.join(self.data_path, "flooding")
        self.normal_path = os.path.join(self.data_path, "normal")

        # Store full paths to images and corresponding labels
        self.flood_data = [os.path.join(self.flood_path, f) for f in os.listdir(self.flood_path)]
        self.normal_data = [os.path.join(self.normal_path, n) for n in os.listdir(self.normal_path)]

        # Combine flood and normal data and create labels
        self.data = self.flood_data + self.normal_data
        self.labels = [1] * len(self.flood_data) + [0] * len(self.normal_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and corresponding label
        img_path = self.data[idx]
        label = self.labels[idx]

        # Open the image as a PIL image
        image = Image.open(img_path).convert('RGB')

        # Apply the transformations (e.g., resize, normalize, etc.)
        if self.transforms:
            image = self.transforms(image)

        # Convert label to a float tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def create_dataloaders(batch_size, transform):
    # Create datasets
    train_set = FloodDataset("./data/train", transform)
    val_set = FloodDataset("./data/valid", transform)
    test_set = FloodDataset("./data/test", transform)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def initialize_model(device):
    # Load a pre-trained MobileNetV2 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    # Replace the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(1280, 1)  # Output size is 1 for binary classification
    )

    # Freeze all the layers except the classifier
    for param in model.features.parameters():
        param.requires_grad = False

    model = model.to(device)
    return model

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure labels have shape [batch_size, 1]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return average_loss, accuracy

def val_one_epoch(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure labels have shape [batch_size, 1]

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Compute accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return average_loss, accuracy

def train_and_validate(model, optimizer, criterion, train_loader, val_loader, device, num_epochs,
                       save_name='training_validation_metrics'):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, criterion, train_loader, device)
        val_loss, val_accuracy = val_one_epoch(model, criterion, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}] - "
              f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.savefig(f"{save_name}_loss_{num_epochs}.png")
    plt.close()

    # Plotting the training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies Over Epochs')
    plt.legend()
    plt.savefig(f"{save_name}_accuracy_{num_epochs}.png")
    plt.close()

    return train_losses, val_losses, train_accuracies, val_accuracies

def compute_test_accuracy(model, test_loader, device, log_file='test_accuracy_log.txt'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Ensure labels have shape [batch_size, 1]

            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Log accuracy to a file
    with open(log_file, 'a') as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

    return accuracy

def run_experiment(num_epochs=10, learning_rate=1e-3, batch_size=32, run_name='run'):
    # Set up device
    device = get_device()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size, transform)

    # Initialize model
    model = initialize_model(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Train and validate
    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(
        model, optimizer, criterion, train_loader, val_loader, device, num_epochs, save_name=f'{run_name}_metrics_plot'
    )

    # Compute test accuracy
    test_accuracy = compute_test_accuracy(model, test_loader, device, log_file=f'{run_name}_test_accuracy_log.txt')

    # Save the model
    model_save_path = f'{run_name}_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_accuracy
    }


if __name__ == "__main__":
    # Example: Run multiple experiments with different settings
    num_runs = 3
    num_epochs = 30
    learning_rates = [5e-4]

    for i in range(num_runs):
        print(f"Starting Run {i + 1} with learning rate {learning_rates[i]}")
        results = run_experiment(
            num_epochs=num_epochs,
            learning_rate=learning_rates[i],
            batch_size=32,
            run_name=f'run_{i + 1}'
        )
