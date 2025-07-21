import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2

# Material class names
class_names = ['Glass', 'Metal', 'Organic', 'Plastic']

# Sustainability score mapping
sustainability_scores = {
    "Plastic": 30,
    "Metal": 60,
    "Organic": 90,
    "Paper": 80
}

# Load trained ResNet-18 model
def load_material_classifier(model_path):
    model = torch.hub.load('pytorch/vision', 'resnet18', weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Predict material from image
def predict_material(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = probs.argmax().item()
        top_class = class_names[top_idx]
        top_confidence = probs[top_idx].item()

    return top_class, top_confidence, probs

# Draw annotated text with background
def draw_text_with_bg(draw, position, text, font, padding=6):
    x, y = position
    text_bbox = draw.textbbox((x, y), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    bg_box = (x - padding, y - padding, x + text_width + padding, y + text_height + padding)
    draw.rectangle(bg_box, fill=(0, 0, 0, 220))
    draw.text((x, y), text, fill="white", font=font)

# Annotate image with predictions
def annotate_image(image_path, material_probs, sustainability_index):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    width, height = image.size
    font_size = max(20, height // 25)
    try:
        font = ImageFont.truetype('arial.ttf', size=font_size)
    except:
        font = ImageFont.load_default()

    y_offset = 10
    for i, material in enumerate(class_names):
        prob = material_probs[i].item()
        text = f"{material}: {prob * 100:.2f}%"
        draw_text_with_bg(draw, (10, y_offset), text, font)
        y_offset += font_size + 10

    text = f"Sustainability Index: {sustainability_index:.2f}"
    draw_text_with_bg(draw, (10, y_offset), text, font)

    annotated_path = "annotated_product.jpg"
    image.save(annotated_path)
    image.show()
    print(f"Annotated image saved as {annotated_path}")

# Webcam capture
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

# File upload
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

# Process image
def process_image():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion("Select Image Source", "Would you like to capture an image from your webcam? (Click 'No' to upload instead)")
    if choice == 'yes':
        image_path = capture_image_from_camera()
    else:
        image_path = select_image_file()

    if not image_path or not os.path.exists(image_path):
        messagebox.showerror("Error", "Invalid or no image file selected.")
        return

    try:
        model = load_material_classifier("models/material_classifier_resnet18.pth")
        material, confidence, all_probs = predict_material(image_path, model)
        sustainability_index = sustainability_scores.get(material, 50)
        annotate_image(image_path, all_probs, sustainability_index)
    except Exception as e:
        messagebox.showerror("Processing Error", str(e))

if __name__ == "__main__":
    process_image()
