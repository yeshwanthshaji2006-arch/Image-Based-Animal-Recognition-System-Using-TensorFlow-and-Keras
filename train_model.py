import json 
import tensorflow as tf 
from tensorflow.keras import layers, models 
 
# ========================== 
# CONFIG 
# ========================== 
DATA_DIR = "animals/animals" 
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32 
EPOCHS = 15 

MODEL_PATH = "animals_MODEL.keras" 
CLASS_NAMES_PATH = "class_names.json" 
 
# ========================== 
# LOAD DATASET 
# ========================== 
# This automatically uses each subfolder name as a class label, 
# and splits 80% train / 20% validation. 
train_ds = tf.keras.utils.image_dataset_from_directory( 
    DATA_DIR, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE 
) 
 
val_ds = tf.keras.utils.image_dataset_from_directory( 
    DATA_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE 
) 
 
class_names = train_ds.class_names 
print("Detected classes (in this exact order):", class_names) 
 
# Save class names so the GUI always matches the model, regardless 
# of folder order on a different machine. 
with open(CLASS_NAMES_PATH, "w") as f: 
 json.dump(class_names, f, indent=2) 
num_classes = len(class_names) 
# Cache + prefetch for faster training 
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) 
# ========================== 
# DATA AUGMENTATION 
# ========================== 
# Helps a lot when you only have a modest number of images per class. 
data_augmentation = models.Sequential([ 
layers.RandomFlip("horizontal"), 
layers.RandomRotation(0.15), 
layers.RandomZoom(0.15), 
layers.RandomContrast(0.1), 
]) 
# ========================== 
# MODEL (Transfer Learning) 
# ========================== 
# MobileNetV2 pretrained on ImageNet gives strong accuracy even with 
# a relatively small fruit dataset, and trains much faster than a 
# from-scratch CNN. 
base_model = tf.keras.applications.MobileNetV2( 
input_shape=IMG_SIZE + (3,), 
include_top=False, 
weights="imagenet" 
) 
base_model.trainable = False  # freeze the pretrained backbone 
inputs = tf.keras.Input(shape=IMG_SIZE + (3,)) 
x = data_augmentation(inputs) 
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) 
x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dropout(0.3)(x) 
outputs = layers.Dense(num_classes, activation="softmax")(x) 
model = tf.keras.Model(inputs, outputs) 
 
model.compile( 
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"] 
) 
 
model.summary() 
 
# ========================== 
# TRAIN 
# ========================== 
early_stop = tf.keras.callbacks.EarlyStopping( 
    monitor="val_accuracy", 
    patience=3, 
    restore_best_weights=True 
) 
 
history = model.fit( 
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS, 
    callbacks=[early_stop] 
) 
 
# ========================== 
# SAVE MODEL 
# ========================== 
model.save(MODEL_PATH) 
print(f"\nSaved model to {MODEL_PATH}") 
print(f"Saved class names to {CLASS_NAMES_PATH}") 
 
final_val_acc = max(history.history["val_accuracy"]) 
print(f"Best validation accuracy: {final_val_acc * 100:.2f}%")