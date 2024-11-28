import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path
from gtts import gTTS
import random
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import easyocr

# Load the model
model = YOLO('/runs/detect/train/weights/best.pt')

def predict_item(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    filename = Path(image_path).stem
    print(filename)

    results = model(image_path, stream=True)

    for i, result in enumerate(results):
        image_filename = f'{output_folder}/{filename}.jpg'
        labels_filename = f'{output_folder}/{filename}.txt'
        print(labels_filename)

        result.save(filename=image_filename)

        if hasattr(result, 'boxes'):
            boxes = result.boxes
            with open(labels_filename, 'w') as f:
                for box in boxes.data:
                    class_id = int(box[5]) 
                    label = result.names[class_id]
                    bbox = tuple(box[:4].cpu().numpy()) 
                    confidence = box[4].item() 
                    f.write(f"{label} {bbox} {confidence:.2f}\n") 

    def split_text_line(line):
        line = line.strip()

        if 'gm' in line:
            unit = 'gm'
        elif 'ml' in line:
            unit = 'ml'
        else:
            return None 
        
        quantity_end_index = line.find(unit) + 2 
        product_name = line[:quantity_end_index].rsplit(' ', 1)[0]
        quantity = line[quantity_end_index-4:quantity_end_index].strip()
        rest = line[quantity_end_index+1:].strip()

        return product_name, quantity, rest
    file_path = labels_filename

    with open(file_path, 'r') as file:
        for line in file:
            product_name, quantity, rest = split_text_line(line)
            print("Product Name:", product_name)
            print("Quantity:", quantity)
            print("The Rest:", rest) 
            return product_name, quantity

# Function to Apply OCR
def apply_easyocr(image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Initialize EasyOCR Reader (Supports multiple languages)
    reader = easyocr.Reader(['en'], gpu=False)

    # Perform OCR
    result = reader.readtext(image_path, detail=0)
    ocr_text = "\n".join(result)

    # Save OCR text
    filename = Path(image_path).stem
    ocr_filename = f"{output_folder}/{filename}_ocr.txt"
    with open(ocr_filename, "w") as f:
        f.write(ocr_text)

    return ocr_text

def main():
    st.set_page_config(page_title="BrailleCart",
                   page_icon = '👓👓',
                   layout = 'centered',
                   initial_sidebar_state = 'collapsed')
    st.title("BrailleCart")

    st.image('/braillecart.webp')

    st.markdown('''<div style="text-align: justify;">
                    See the Unseen: Empowering vision, Enhancing individual lives with AI.
                </div>''', unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_path = os.path.join("temp", uploaded_image.name)
        os.makedirs("temp", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Load the image using PIL to display (optional)
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=500)

        output_folder = 'output'

        if st.button('Predict'):
            with st.spinner('Predicting...'):
                predicted_class, predicted_quantity = predict_item(image_path, output_folder)
                st.success(f"YOLO Prediction: {predicted_class}")
                # Apply OCR
                ocr_text = apply_easyocr(image_path, output_folder)
                st.success(f"OCR Prediction: {ocr_text}")
                print(ocr_text)
                if predicted_class:
                    response = generate_detailed_description(predicted_class, predicted_quantity)
                    st.success(f"Meta Llama 3.2 Response: {response}")
                    convert_to_speech(response)
                else:
                    response = generate_detailed_description_ocr(ocr_text)
                    st.success(f"Meta Llama 3.2 Response: {response}")
                    convert_to_speech(response)

            if os.path.exists("temp"):
                shutil.rmtree("temp")