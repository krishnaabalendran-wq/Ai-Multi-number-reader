import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Deeper layers with Batch Normalization for better feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), #input layer
            nn.BatchNorm2d(32), #stabilize
            nn.ReLU(), #complex pattern
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # small changes
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # NEW Block 3: Adding 128 filters for even more complex feature extraction
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25) #Prevent Overfitting
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512), #makes the inputs turn into 512 neurons
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5), #Prevent Overfitting
            nn.Linear(512, 10) #those 512 neurons become digits 0-9
        )

    def forward(self, x):
        #Image is passed through as input and gives the predictions
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def train_engine(model, loader, optimizer, loss_fn, device, epoch):
    #Setting the model to train so it can use dropout.
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # No bias from older batches
        output = model(data)
        loss = loss_fn(output, target) 
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} | Loss: {loss.item():.6f}")

def predict_line_by_line_gui(model, image_path, device, stats):
    # Load image
    img = cv2.imread(image_path) 
    if img is None:
        print("Error: Image not found.")
        return
    #Cleaning data by removing color, cleaning noise, seperate background to digits and adding thikness for digits
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    #finds Outlines and then filters out more noise
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 5 and h > 10:
            boxes.append([x, y, w, h])
    
    if not boxes:
        print("No digits found.")
        return

    #To read multiple lines the model must determine if the distance of numbers is large from each line
    boxes.sort(key=lambda b: b[1])

    lines = []
    current_line = [boxes[0]]
    for i in range(1, len(boxes)):
        if boxes[i][1] > (current_line[-1][1] + current_line[-1][3] * 0.7):
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [boxes[i]]
        else:
            current_line.append(boxes[i])
    lines.append(sorted(current_line, key=lambda b: b[0]))

    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    processed_lines = []
    max_digits_in_line = 0

    for line in lines:
        valid_line_data = []
        for x, y, w, h in line:
            size = max(w, h) + 20 # padding
            pad_img = np.zeros((size, size), dtype="uint8") # creates blank area of black
            #Grabs the center of the rendered image and moves it to the center of black area
            pad_img[(size-h)//2 : (size-h)//2+h, (size-w)//2 : (size-w)//2+w] = thresh[y:y+h, x:x+w] 
            #Trained in 28x28 so resizes it for model to scan
            roi = cv2.resize(pad_img, (28, 28), interpolation=cv2.INTER_AREA)
            #Convert from 0-1, normalizes and then makes it into a list of one
            img_tensor = transform(Image.fromarray(roi)).unsqueeze(0).to(device)
            #Guessing and outputs confidence so must be greater than 50 to remove any more noise that was not detected.
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                # Confidence gate
                if conf.item() >= 0.50:
                    valid_line_data.append({'roi': roi, 'pred': pred.item(), 'conf': conf.item()})
                    stats['total_tested'] += 1
                    stats['total_correct'] += 1
        
        if valid_line_data:
            processed_lines.append(valid_line_data)
            max_digits_in_line = max(max_digits_in_line, len(valid_line_data))

    if not processed_lines:
        print("No digits passed the 50% confidence threshold.")
        return

    #Size of matplotlib gui
    num_lines = len(processed_lines)
    fig = plt.figure(figsize=(max_digits_in_line * 2, num_lines * 2))
    
    full_output_string = []
    #Table grid
    for r, line_data in enumerate(processed_lines):
        line_str = ""
        for c, data in enumerate(line_data):
            ax = plt.subplot2grid((num_lines, max_digits_in_line), (r, c))
            #Predictions to show up and uploads an image of what the model scanned
            ax.imshow(data['roi'], cmap='gray')
            ax.set_title(f"L{r+1} | P:{data['pred']}\n{data['conf']*100:.1f}%", fontsize=8)
            ax.axis('off')
            line_str += str(data['pred'])
        full_output_string.append(line_str)
    #Formatting
    display_text = " | ".join(full_output_string)
    fig.suptitle(f"Readout: {display_text}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()
    #Terminal work
    for i, s in enumerate(full_output_string):
        print(f"Line {i+1}: {s}")

if __name__ == '__main__':
    # Improved device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Robust data augmentation for real-world scans
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=train_transform), 
                              batch_size=128, shuffle=True)

    model = ConvNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Learning rate scheduler to fine-tune accuracy in later epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    print(f"Training on {device}...")
    
    for epoch in range(1, 7):
        train_engine(model, train_loader, optimizer, nn.CrossEntropyLoss(), device, epoch)
        scheduler.step()
    
    session_stats = {'total_tested': 0, 'total_correct': 0}

    prompt=True
    while prompt:
        prompt = input("Enter image address: ")
        predict_line_by_line_gui(model, prompt, device, session_stats)
