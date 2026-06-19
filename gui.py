import json 
import tkinter as tk 
from tkinter import filedialog 
 
import numpy as np 
import tensorflow as tf 
from PIL import Image, ImageTk 
 
# ========================== 
# LOAD MODEL + CLASS NAMES 
# ========================== 
MODEL_PATH = "animals_MODEL.keras" 
CLASS_NAMES_PATH = "class_names.json" 
 
model = tf.keras.models.load_model(MODEL_PATH) 
 
# Load the class names that were saved during training, so the 
# label order always matches what the model was actually trained on 
# (instead of a hardcoded list that can silently go out of sync). 
with open(CLASS_NAMES_PATH, "r") as f: 
    class_names = json.load(f) 
 
# Read the input size the model expects, so this still works even if 
# you retrain with a different IMG_SIZE later. 
IMG_SIZE = model.input_shape[1:3]  # (height, width) 
 
 
# ========================== 
# PREDICT FUNCTION 
# ========================== 
def predict_image(): 
 
    file_path = filedialog.askopenfilename( 
        filetypes=[ 
            ("Image Files", "*.jpg *.png *.jpeg") 
        ] 
    ) 
 
    if not file_path: 
        return 
 
    # Display image in the GUI 
    img = Image.open(file_path).convert("RGB") 
    display_img = img.copy() 
    display_img.thumbnail((300, 300)) 
 
    photo = ImageTk.PhotoImage(display_img) 
    image_label.config(image=photo) 
    image_label.image = photo 
 
    # Prepare image for the model 
    img_for_model = img.resize(IMG_SIZE) 
    img_array = np.array(img_for_model, dtype=np.float32) 
 
    # NOTE: the training script uses MobileNetV2's own preprocessing 
    # (which the model performs internally), so we feed raw 0-255 
    # pixel values here rather than dividing by 255. 
    img_array = np.expand_dims(img_array, axis=0) 
 
    prediction = model.predict(img_array) 
    class_index = int(np.argmax(prediction)) 
    confidence = float(np.max(prediction) * 100) 
 
    result_label.config( 
        text=f"Prediction : {class_names[class_index]}" 
    ) 
    confidence_label.config( 
        text=f"Confidence : {confidence:.2f}%" 
    ) 
 
 
# ========================== 
# GUI 
# ========================== 
root = tk.Tk() 
root.title("Animal detection using AI") 
root.geometry("800x600") 
 
title = tk.Label( 
    root, 
    text="Animal Detection Using AI", 
    font=("Arial", 20, "bold") 
) 
title.pack(pady=10) 
 
image_label = tk.Label(root) 
image_label.pack(pady=20) 
 
btn = tk.Button( 
    root, 
    text="Select Image", 
    font=("Arial", 14), 
    command=predict_image 
) 
btn.pack() 
 
result_label = tk.Label( 
    root, 
    text="Prediction : ", 
    font=("Arial", 18, "bold") 
) 
result_label.pack(pady=20) 
 
confidence_label = tk.Label( 
    root, 
    text="Confidence : ", 
    font=("Arial", 16) 
) 
confidence_label.pack() 
 
root.mainloop() 