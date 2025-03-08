import os
os.system("git lfs pull")
import os
os.system('pip install pillow')
from PIL import Image
# Your code here
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")
from ultralytics import YOLO
import logging
import datetime
import gradio as gr
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image, ImageDraw
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load the model (assuming 'best.pt' contains a suitable model)
try:
    model = YOLO('best.pt')
except FileNotFoundError:
    print("Model file 'best.pt' not found. Please upload it.")
except Exception as e:
    print(f"Error loading the model: {e}")

def create_pdf_report(Date,name, gender, age, result, image_path, pdf_path):
    """Creates a PDF report with the given result and patient particulars."""
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Brain Tumor Detection Report")

    # Add patient details and prediction result
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, f"Date of Report: {Date}")
    c.drawString(100, height - 150, f"Patient Name: {name}")
    c.drawString(100, height - 170, f"Gender: {gender}")
    c.drawString(100, height - 190, f"Age: {age}")
    c.drawString(100, height - 210, f"Result: {result}")

    # Add the MRI image
    c.drawImage(image_path, 130, height - 550, width=300, height=300)
     # Get current date
    

    c.save()

def predict_brain_tumor(image, name, gender, age):
    """Predicts brain tumor presence in an image and draws bounding boxes."""
    if model is None:
        return "Model loading failed. Please check the model file.", None

    try:
        # Convert image to numpy array if required
        image_array = np.array(image)

        # Perform the prediction
        prediction = model(image_array)  # Use YOLO's prediction method
        logging.debug(f"Prediction output: {prediction}")

        # Extract results and draw bounding boxes
        result = "No brain tumor detected"
        if len(prediction[0].boxes) > 0:  # Check if there are any bounding boxes
            result = "Brain tumor detected"
            annotated_image = np.array(image)  # Start with original image

            for box in prediction[0].boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                conf = box.conf[0]  # Confidence score
                label = f"{prediction[0].names[0]} ({conf:.2f})"

                # Draw bounding box on image
                annotated_image = cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                annotated_image = cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5, (255, 0, 0), 2)

            # Save the annotated image
            image_path = "/tmp/annotated_image.png"
            Image.fromarray(annotated_image).save(image_path)
        else:
            # Save the input image (unaltered)
            image_path = "/tmp/input_image.png"
            image.save(image_path)

        # Save the Date
        Date = datetime.date.today().strftime("%Y-%m-%d")

        # Create the PDF report
        pdf_path = "/tmp/Brain_Tumor_Detection_Report.pdf"
        create_pdf_report(Date, name, gender, age, result, image_path, pdf_path)

        return f"Date of Report: {Date}, Patient Name: {name}, Gender: {gender}, Age: {age}. Result: {result}", pdf_path

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return f"Prediction error: {e}", None
# Create the Gradio interface
iface = gr.Interface(
    fn=predict_brain_tumor,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Patient Name"),
        gr.Radio(choices=["Male", "Female", "Other"], label="Gender"),
        gr.Number(label="Age")
    ],
    outputs=["text", "file"],
    title="Brain Tumor Detection Report",
    description="Upload MRI scan to detect brain tumors.",
)

# Launch the interface
iface.launch(share=True) 