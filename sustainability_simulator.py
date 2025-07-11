import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import time

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

# Annotate with Background
def draw_text_with_bg(draw, position, text, font, padding=6):
    x, y = position

    text_bbox = draw.textbbox((x, y), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    bg_box = (
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding
    )
    draw.rectangle(bg_box, fill=(0, 0, 0, 220))

    draw.text((x, y), text, fill="white", font=font)

# Annotate Image with Predictions
def annotate_image(image_path, material_probs, sustainability_index):
    # Load Image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Dynamically scale font
    width, height = image.size
    font_size = max(20, height // 25)
    try:
        font = ImageFont.truetype('arial.ttf', size=font_size)
    except:
        font = ImageFont.load_default()
    
    # Annotate Image
    y_offset = 10
    for i, material in enumerate(['Plastic', 'Metal', 'Organic']):
        text = f"{material}: {material_probs[i]*100:.2f}"
        draw_text_with_bg(draw, (10, y_offset), text, font)
        y_offset += font_size + 14

    text = f"Sustainability Index: {sustainability_index:.2f}"
    draw_text_with_bg(draw, (10, y_offset), text, font)

    # Save Annotated Image
    annotated_path = "annotated_product.jpg"
    image.save(annotated_path)
    image.show()
    print(f"Annotated image saved as {annotated_path}")

# Capture Image
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return None

    messagebox.showinfo("Info", "Press 's' to capture image, or 'q' to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capture Image (press s to save, q to quit)', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            captured_path = 'captured_product.jpg'
            cv2.imwrite(captured_path, frame)
            break
        elif key == ord('q'):
            captured_path = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_path

# Select Image File
def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Product Image",
        filetypes=[
            ('Image files', '*.jpg *.jpeg *.png *.webp'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('WebP files', '*.webp')
        ]
    )
    return file_path

# Main Processing
def process_image():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion("Select Image Source", "Would you like to capture an image from your webcam? (Click 'No' to upload instead)")

    # Load Image
    if choice == 'yes':
        image_path = capture_image_from_camera()
    else:
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
