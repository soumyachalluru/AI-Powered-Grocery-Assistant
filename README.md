# BrailleCart: AI-Powered Grocery Assistance for the Visually Impaired

**BrailleCart** is an AI-powered system designed to assist visually impaired individuals in identifying and learning about grocery items in real-time. Combining advanced AI technologies like YOLOv8n for object detection, OCR for extracting product details, and LLaMA-based conversational text-to-speech, BrailleCart enhances the shopping experience through a simple and accessible interface.

---

## Dataset
[Dataset download link](https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2/dataset/5)

## Key Features
- **Real-Time Object Detection**: Utilizes YOLOv8n for precise and quick identification of grocery items.
- **Text Recognition (OCR)**: Extracts product names, prices, and other details from labels.
- **Conversational Feedback**: Uses LLaMA to generate natural language descriptions of detected items.
- **Audio Accessibility**: Provides real-time audio feedback with Google Text-to-Speech (gTTS).
- **User-Friendly Interface**: Streamlit-based UI for seamless interaction.

---

## Technologies Used
- **YOLOv8n**: Lightweight and accurate object detection.
- **EasyOCR**: Reliable Optical Character Recognition for extracting text from images.
- **LLaMA (Large Language Model)**: Generates accessible and user-friendly descriptions.
- **Streamlit**: Simplifies user interaction with the system.
- **Google Text-to-Speech (gTTS)**: Converts text descriptions into audio output.

---

## How It Works
1. **Object Detection**: YOLOv8n identifies grocery items and provides bounding boxes and labels.
2. **Text Extraction**: EasyOCR reads labels for additional details like price and weight.
3. **Natural Language Descriptions**: LLaMA generates concise, context-aware audio descriptions.
4. **Audio Feedback**: gTTS converts text to speech for accessibility.
5. **User Interaction**: Users can upload images or use a live camera feed to interact with the system.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/soumyachalluru/AI-Powered-Grocery-Assistant.git
   cd BrailleCart

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Start the application:
    ```bash
    streamlit run app.py

4. Upload images or enable the live camera feed to detect grocery items and receive audio feedback.
