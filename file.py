# 1. Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# 2. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Define Transformations (Normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4. Download and Load MNIST Dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# 5. Visualize Some Samples
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.figure(figsize=(8, 8))
for i in range(30):
    idx = random.randint(0, len(example_data) - 1)
    plt.subplot(5, 6, i + 1)
    plt.imshow(example_data[idx][0], cmap='gray')
    plt.title(f"Label: {example_targets[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Shape of a single image:", example_data[0].shape)
print("Shape of a batch:", example_data.shape)

# 6. Define the CNN Model
class DigitCNN(nn.Module):
    def _init_(self, num_classes=10):
        super(DigitCNN, self)._init_()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# 7. Initialize Model, Loss Function, Optimizer
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
epochs = 10


# 8. Training and Validation Loop
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

     # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_correct / val_total)

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]*100:.2f}% | "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]*100:.2f}%")
    

    # 9. Final Test Evaluation
model.eval()
test_loss, test_correct, test_total = 0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

final_test_loss = test_loss / len(test_loader)
final_test_accuracy = test_correct / test_total

print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_accuracy*100:.2f}%")

# 10. Visualize Predictions
classes = [str(i) for i in range(10)]

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
    plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Save the trained model
torch.save(model.state_dict(), "digit_cnn.pth")
print("Model saved successfully!")

# Re-create the model architecture
loaded_model = DigitCNN().to(device)
loaded_model.load_state_dict(torch.load("digit_cnn.pth"))
loaded_model.eval()

print("Model loaded successfully!")

from google.colab import files
from PIL import Image
import io
import torchvision.transforms as transforms

# Upload an image
uploaded = files.upload()

# Get the uploaded image name
img_name = list(uploaded.keys())[0]


# Load and preprocess the image
def preprocess_image(img_path):
    image = Image.open(img_path)

 # Convert to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')  # 'L' mode means 8-bit pixels, black and white

    transform = transforms.Compose([
        transforms.Resize((28, 28)),                # Resize to 28x28
        transforms.ToTensor(),                      # Convert to Tensor
        transforms.Normalize((0.1307,), (0.3081,))   # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    return image.to(device)


# Predict function
def predict(image_tensor):
    with torch.no_grad():
        output = loaded_model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()


# Use the functions
image_tensor = preprocess_image(img_name)
prediction = predict(image_tensor)

print(f"Predicted Digit: {prediction}")

