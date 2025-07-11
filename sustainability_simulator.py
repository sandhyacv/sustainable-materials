import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Simulated Model for Demonstration
class SimulatedSustainabilityModel(torch.nn.Module):
    def __init__(self):
        super(SimulatedSustainabilityModel, self).__init__()
        self.materials = ['Plastic', 'Metal', 'Organic']
        self.num_materials = len(self.materials)
    
    def forward(self, x):
        # Simulate material composition prediction 
        material_probs = torch.softmax(torch.randn(self.num_materials), dim=0)
        # Simulate sustainability index prediction
        sustainability_index = torch.sigmoid(torch.randn(1)) * 100
        return material_probs, sustainability_index.item()

# Annotate Image with Predictions
def annotate_image(image_path, material_probs, sustainability_index):
    # Load Image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype('arial.ttf', size=20)
    except:
        font = ImageFont.load_default()
    
    # Annotate Image
    y_offset = 10
    for i, material in enumerate(['Plastic', 'Metal', 'Organic']):
        text = f"{material}: {material_probs[i]*100:.2f}"
        draw.text((10, y_offset), text, fill='white', font=font)
        y_offset += 30
    
    text = f"Sustainability Index: {sustainability_index:.2f}"
    draw.text((10, y_offset), text, fill='white', font=font)

    # Save Annotated Image
    annotated_path = "annotated_product.jpg"
    image.save(annotated_path)
    image.show()
    print(f"Annotated image saved as {annotated_path}")

# Select Image File
def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Product Image",
        filetypes=[('JPEG files', '*.jpg *.jpeg'), ('PNG files', '*.png')]
    )
    return file_path

# Main Processing
def process_image():
    # Load Image
    image_path = select_image_file()
    if not image_path or not os.path.exists(image_path):
        messagebox.showerror("Error", "Invalid or no image file selected.")
        return

    # Preprocess Image
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        model = SimulatedSustainabilityModel()
        material_probs, sustainability_index = model(image_tensor)

        annotate_image(image_path, material_probs, sustainability_index)
    except Exception as e:
        messagebox.showerror("Processing Error", str(e))

# Run
if __name__ == "__main__":
    process_image()