import streamlit as st
import sys
import os

# Add error handling for imports
try:
    import torch
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

try:
    import google.generativeai as genai
except ImportError:
    st.error("Google Generative AI is not installed. Please install it using 'pip install google-generativeai'")
    st.stop()

# Configuration and Constants
DISEASE_CONTEXT = """
Apple Leaf Diseases Overview:
1. Apple Scab:
   - Caused by Venturia inaequalis fungus
   - Symptoms: Dark, scabby lesions on leaves
   - Thrives in cool, wet conditions
   - Most common in spring and early summer

2. Black Rot:
   - Caused by Botryosphaeria obtusa fungus
   - Symptoms: Circular brown spots with dark borders
   - Progresses from leaf to fruit
   - Common in warm, humid regions

3. Cedar Apple Rust:
   - Caused by Gymnosporangium juniperi-virginianae
   - Symptoms: Yellow-orange spots on leaves
   - Requires alternate hosts (cedar/juniper trees)
   - Most prevalent in areas with both apple and cedar trees
"""

class AppleLeafDiseaseDetector:
    def __init__(self, yolo_model_path, llm_api_key):
        try:
            # Initialize YOLO model
            self.yolo_model = YOLO(yolo_model_path)
            
            # Configure Google Generative AI
            genai.configure(api_key=llm_api_key)
            self.llm_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            raise

    def detect_disease(self, image, prompt):
        try:
            # Perform YOLO detection
            results = self.yolo_model(image)

            # Create a copy of the image for annotation
            annotated_image = image.copy()
            
            # Prepare context for LLM
            context = f"{DISEASE_CONTEXT}\n\nUser Input Description: {prompt}"
            
            # Prepare visual detection information
            detection_info = "\nDetection Results:\n"
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name
                    cls_idx = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_idx]
                    conf = float(box.conf[0])
                    detection_info += f"- {class_name} detected with confidence {conf:.2f}\n"

                    # Get box coordinates; assuming they are in xyxy format
                    # Note: .cpu().numpy() is used to ensure compatibility if using GPU
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    # Draw rectangle on the annotated image
                    # Make thickness and font size relative to image size
                    thickness = int(min(annotated_image.shape[0], annotated_image.shape[1]) / 100)
                    font_size = min(annotated_image.shape[0], annotated_image.shape[1]) / 1000
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(99, 49, 222), thickness=thickness)
                    # Put class name text near the top-left corner of the bounding box
                    cv2.putText(annotated_image, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (99, 49, 222), thickness//2)
            
            # Full prompt for LLM
            full_prompt = f"{context}\n{detection_info}\n\nBased on the detection and description, diagnose the specific disease and provide management recommendations."
            
            # Generate response using LLM
            response = self.llm_model.generate_content(full_prompt)
            
            return detection_info, response.text, annotated_image
        except Exception as e:
            st.error(f"Detection Error: {e}")
            return None, None, None

def main():
    st.title("Apple Leaf Disease Detector")
    
    
    # Image and description inputs
    uploaded_image = st.file_uploader("Upload Leaf Image", type=['png', 'jpg', 'jpeg'])
    description = st.text_area("Describe Leaf Condition", 
                                placeholder="Include weather, season, location, and observed symptoms")
    
    # Validate inputs and perform detection
    if uploaded_image and description:
        try:
            # Read image
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Initialize detector
            detector = AppleLeafDiseaseDetector("last.pt", "AIzaSyDYOMW3owdKqT8obveTemCqplHA08X6CPg")
            
            # Detect and diagnose
            if st.button("Analyze Leaf"):
                with st.spinner('Analyzing...'):
                    detection_info, diagnosis, annotated_image = detector.detect_disease(image, description)
                    
                    if detection_info and diagnosis:
                        # Display results
                        st.subheader("Detection Results")
                        st.text(detection_info)

                        # Display the annotated image
                        st.subheader("Annotated Image")
                        # Convert BGR (OpenCV default) to RGB for correct color display in Streamlit
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image_rgb, caption="YOLO Detection", use_container_width=True)
                        
                        st.subheader("Disease Diagnosis")
                        st.write(diagnosis)
                    else:
                        st.error("Failed to analyze the image. Please check your inputs.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()