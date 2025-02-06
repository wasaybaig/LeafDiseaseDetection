import streamlit as st
import sys
import os

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

DISEASE_CONTEXT = """
Apple Leaf Diseases Overview:
# 1. Apple Scab
## Symptoms & Identification
Leaf: Gray-brown or black velvety spots with circular or radial lesions, often leading to yellowing and premature defoliation.

Fruit: Cracked, scabby lesions that reduce quality.

## Major Cause
Fungus Venturia inaequalis spreads via rain-splashed spores from infected leaves or fruit.

## Susceptible Regions/Climate

Climate: Cool (10–24°C), wet springs with prolonged leaf wetness.

Regions: Temperate zones (e.g., North America, Europe, parts of Asia).

## Management

Chemical: Apply fungicides (e.g., sulfur, copper) during bud break.

Cultural: Plant resistant cultivars (e.g., ‘Liberty’), remove fallen leaves, and prune for airflow.

# 2. Cedar Apple Rust
## Symptoms & Identification

Leaf: Bright orange/yellow spots with red borders; gelatinous spore-producing structures (telial horns) on undersides.

Fruit: Distorted or stunted growth with rust-colored lesions.

## Major Cause

Fungus Gymnosporangium juniperi-virginianae requires two hosts: apple trees and junipers/cedars.

## Susceptible Regions/Climate

Climate: Humid, moderate temperatures (18–25°C) during spring

Regions: Areas with juniper/cedar populations (e.g., eastern North America, East Asia).

## Management

Chemical: Fungicides (e.g., myclobutanil) applied during early spring.

Cultural: Remove juniper hosts within 1 km of orchards; use rust-resistant varieties (e.g., ‘Redfree’).

# 3. Black Rot
## Symptoms & Identification

Leaf: Dark brown, concentric "frogeye" spots with purple margins; severe cases cause leaf curling.

Fruit: Rotting with concentric rings and mummification,

## Major Cause

Fungus Botryosphaeria obtusa infects through wounds or stressed tissues.

## Susceptible Regions/Climate

Climate: Warm (25–30°C), humid summers with frequent rain.

Regions: Global in apple-growing areas (e.g., North America, China, Europe).

## Management

Chemical: Apply captan or thiophanate-methyl during flowering.
Cultural: Prune dead branches, remove mummified fruit, and avoid mechanical damage to trees.
"""

class AppleLeafDiseaseDetector:
    def __init__(self, yolo_model_path, llm_api_key):
        try:
            self.yolo_model = YOLO(yolo_model_path)
            
            genai.configure(api_key=llm_api_key)
            self.llm_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Initialization Error: {e}")
            raise

    def detect_disease(self, image, prompt):
        try:
            # Perform YOLO detection
            results = self.yolo_model(image)

            annotated_image = image.copy()
            
            context = f"{DISEASE_CONTEXT}\n\nUser Input Description: {prompt}"
            
            detection_info = "\nDetection Results:\n"
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_idx = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_idx]
                    conf = float(box.conf[0])
                    detection_info += f"- {class_name} detected with confidence {conf:.2f}\n"

                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    thickness = int(min(annotated_image.shape[0], annotated_image.shape[1]) / 100)
                    font_size = min(annotated_image.shape[0], annotated_image.shape[1]) / 1000
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(99, 49, 222), thickness=thickness)
                    cv2.putText(annotated_image, class_name, (x1, y1 - 5*thickness),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (99, 49, 222), thickness//2)
            
            full_prompt = f"You are a helpful assistant to diagnose apple leaf disease. Based on the detection and description, diagnose the specific apple leaf disease and provide management recommendations to the user. \nContext: \n{context}\n Detection Info: \n{detection_info}"
            
            response = self.llm_model.generate_content(full_prompt)
            
            return detection_info, response.text, annotated_image
        except Exception as e:
            st.error(f"Detection Error: {e}")
            return None, None, None

def main():
    st.title("Apple Leaf Disease Detector")
    
    
    uploaded_image = st.file_uploader("Upload Leaf Image", type=['png', 'jpg', 'jpeg'])
    description = st.text_area("Describe Leaf Condition", 
                                placeholder="Include weather, season, location, and observed symptoms")
    
    if uploaded_image and description:
        try:
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            
            detector = AppleLeafDiseaseDetector("last.pt", "AIzaSyDYOMW3owdKqT8obveTemCqplHA08X6CPg")
            
            if st.button("Analyze Leaf"):
                with st.spinner('Analyzing...'):
                    detection_info, diagnosis, annotated_image = detector.detect_disease(image, description)
                    
                    if detection_info and diagnosis:
                        # Display results
                        st.subheader("Detection Results")
                        st.text(detection_info)

                        # Display the annotated image
                        st.subheader("Annotated Image")
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
