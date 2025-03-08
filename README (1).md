# Brain-Tumor-Cells-Detection
This project will detect tumor in brain MRI scans to help assist the radiologist/expert in detecting tumor accurately and within less time.\
*1st Step: Collection of dataset*
The dataset has been taken from Roboflow \
            Go to Roboflow Universe or follow the link\
            ![Roboflow](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/roboflow_logo.png)\
            (Link:https://universe.roboflow.com/zaky-indra-w86c4/brain-tumor-gsh0d/dataset/1).\
          ** Perticulars of dataset**
            MRI Scans ~3073\
            Train:2766 (90%)\
            Valid:184 (6%)\
            Test:123 (4%)\
            Preprocesing :Auto-Orient: Applied\
            Resize: Stretch to 640x640\
            Output per training example: 3\
            Grayscale: Apply to 15% of images\
            Blur: Up to 1px\
            Noise: Up to 1.49% of pixels\
            Bounding Box: Flip: Horizontal\
![image](https://github.com/user-attachments/assets/aaac2cbb-3aab-46d5-91c0-5b0c84161222)\
           Click the download dataset.The dataset will be downloaded in downloads.\
**2nd Step** Go to colab (https://colab.research.google.com/#create=true)\
**3rd Step** Upload data set to google drive \
**4th step** Import dataset mount drive using \
**5th step** unzip data using following code: import zipfile # This line imports the zipfile module\
              zip_path = '/content/path'\
              extract_path = 'path'\
              with zipfile.ZipFile(zip_path, 'r') as zip_ref:\
                  zip_ref.extractall(extract_path)\
**6th step** setting up environment\
              Import necessary libraries \
              import pandas as pd\
              import seaborn as sns\
              import matplotlib.pyplot as plt\
              import numpy as np\
              import os,random\
              import matplotlib.pyplot as plt\
              import matplotlib.image as mpimg\
              import cv2\
              import torch\
              !pip install Pillow\
              import os\
              from PIL import Image\
              import os\
              from PIL import Image\
              import ray\
              print(ray.__version__)\
              All major libraries and packages have also been listed in the dependencies file.\
  **7th step**
              Covert images from grey to rgb scale using defining a function 
              def convert_images_to_rgb(directory):
                  for root, _, files in os.walk(directory):
                      for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                              image_path = os.path.join(root, file)
                              try:
                              img = Image.open(image_path)
                              img_rgb = img.convert('RGB')
                              img_rgb.save(image_path)
                              print(f"Converted {file} to RGB")
                      except Exception as e:\
                              print(f"Warning: Could not convert {file} to RGB. Error: {e}")
 **8th step** Ceate Image directory and display images in dataset to get verification\
            (Write the script in colab like this)\
            image_directory = '/content/drive/MyDrive/Roboflow_Brain_Tumor/brain tumor.v1i.yolov8/train/images'\
            num_sample = 9\
            image_files = os.listdir(image_directory)\
            random_images = random.sample(image_files,num_sample)\
            fig ,axes = plt.subplots(3,3,figsize=(11,11))\
            for i in range(num_sample):\
              ax = axes[i//3,i%3]\
              ax.imshow(mpimg.imread(os.path.join(image_directory,random_images[i])))\
              ax.axis('off')\
              ax.set_title(f'Image {i}')\
              plt.tight_layout()\
            plt.show()\
![Images verification](https://github.com/user-attachments/assets/2ca23625-454d-4ed1-8e04-2444f979a8ab)\
**Step 9** # shape of single image\
            image_dr = os.path.join(image_directory,random_images[0])
            image_dr
**Step 10** Download YOLOv8 and run Inference on a random image\
                    from ultralytics import YOLO\
             Write scrite in cloab\
             yolo_model = YOLO('yolov8n.pt')\
                 (To check with inference write this scipte in colab)\
                 !yolo task=detect mode=predict model=yolov8n.pt source='/content/drive/MyDrive/Roboflow_Brain_Tumor/brain             tumor.v1i.yolov8/train/images/y424_jpg.rf.35c1baa4ae87ff74e91ecaf4a0c21537.jpg'\
**Step 11** Train model on custom dataset \
            Model has been given epoch 16, batch size=-1, optimiser=auto\
            use following script\
Result_Final_model2 = yolo_model.train(data="/content/drive/MyDrive/Roboflow_Brain_Tumor/brain tumor.v1i.yolov8/data.yaml",epochs = 16,batch =-1, optimizer = 'auto')\
**Step 12** Check Model Accuracy\
Few metrics are\
            confusion matrix\
            F1 curve\
            Precision and confidence score\
            Recall and confidence score\
            ![Recall for model](\
**Step 13** The results of the model are saved in form of csv file That can be visualize as\
Firstly reading dataframe\
Result_Train_model12 = pd.read_csv('/content/runs/detect/train/results.csv')\
Afterwards displaying only 10 last rows of that\
Result_Train_model12.tail(10)\
Also wrinting following script\
 Read the results.csv file as a pandas dataframe\
Result_Train_model.columns = Result_Train_model.columns.str.strip()

# Create subplots
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))\

# Plot the columns using seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=Result_Train_model, ax=axs[0,0])\
sns.lineplot(x='epoch', y='train/cls_loss', data=Result_Train_model, ax=axs[0,1])\
sns.lineplot(x='epoch', y='train/dfl_loss', data=Result_Train_model, ax=axs[1,0])\
sns.lineplot(x='epoch', y='metrics/precision(B)', data=Result_Train_model, ax=axs[1,1])\
sns.lineplot(x='epoch', y='metrics/recall(B)', data=Result_Train_model, ax=axs[2,0])\
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=Result_Train_model, ax=axs[2,1])\
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=Result_Train_model, ax=axs[3,0])\
sns.lineplot(x='epoch', y='val/box_loss', data=Result_Train_model, ax=axs[3,1])\
sns.lineplot(x='epoch', y='val/cls_loss', data=Result_Train_model, ax=axs[4,0])\
sns.lineplot(x='epoch', y='val/dfl_loss', data=Result_Train_model, ax=axs[4,1])\

# Set titles and axis labels for each subplot
axs[0,0].set(title='Train Box Loss')\
axs[0,1].set(title='Train Class Loss')\
axs[1,0].set(title='Train DFL Loss')\
axs[1,1].set(title='Metrics Precision (B)')\
axs[2,0].set(title='Metrics Recall (B)')\
axs[2,1].set(title='Metrics mAP50 (B)')\
axs[3,0].set(title='Metrics mAP50-95 (B)')\
axs[3,1].set(title='Validation Box Loss')\
axs[4,0].set(title='Validation Class Loss')\
axs[4,1].set(title='Validation DFL Loss')\


plt.suptitle('Training Metrics and Loss', fontsize=24)\
plt.subplots_adjust(top=0.8)\
plt.tight_layout()\
plt.show()\
This will produce following results\
![Results]()\
**Step 14**
Yolo automatically stores best performing model in train/weights/best.pt. So, utilizing that model evaluation on valid set was done using following script. and results may e printed\
# Loading the best performing model
Valid_model = YOLO('/content/runs/detect/train/weights/best.pt')\
**To save this model download this model **\
# Evaluating the model on the validset
metrics = Valid_model.val(split = 'val')
![Validation dataset inference](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/download.png)
# final results
print("precision(B): ", metrics.results_dict["metrics/precision(B)"])\
print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])\
print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])\
print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])\
Model produce following results\
Results saved to runs/detect/val\
precision(B):  0.9323591286180969\
metrics/recall(B):  0.9189348599890331\
metrics/mAP50(B):  0.9668363585639872\
metrics/mAP50-95(B):  0.7805349340372879\
**Step 15**
# Normalization function
def normalize_image(image):
    return image / 255.0

# Image resizing function
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Path to test images
dataset_path = '/content/drive/MyDrive/Roboflow_Brain_Tumor/brain tumor.v1i.yolov8'  # Place your dataset path here
valid_images_path = os.path.join(dataset_path, 'valid', 'images')

# List of all jpg images in the directory
image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]

# Check if there are images in the directory
if len(image_files) > 0:
    # Select 9 images at equal intervals
    num_images = len(image_files)
    step_size = max(1, num_images // 9)  # Ensure the interval is at least 1
    selected_images = [image_files[i] for i in range(0, num_images, step_size)]

    # Prepare subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('val Set Inferences', fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])

            # Load image
            image = cv2.imread(image_path)

            # Check if the image is loaded correctly
            if image is not None:
                # Resize image
                resized_image = resize_image(image, size=(640, 640))
                # Normalize image
                normalized_image = normalize_image(resized_image)

                # Convert the normalized image to uint8 data type
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

                # Predict with the model
                results = Valid_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)

                # Plot image with labels
                annotated_image = results[0].plot(line_width=1)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ax.imshow(annotated_image_rgb)
            else:
                print(f"Failed to load image {image_path}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    ![Valid set Images](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/download.png)

**Step 16**
# Normalization function
def normalize_image(image):
    return image / 255.0

# Image resizing function
def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# Path to test images
dataset_path = '/content/drive/MyDrive/Roboflow_Brain_Tumor/brain tumor.v1i.yolov8'  # Place your dataset path here
valid_images_path = os.path.join(dataset_path, 'test', 'images')

# List of all jpg images in the directory
image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]

# Check if there are images in the directory
if len(image_files) > 0:
    # Select 9 images at equal intervals
    num_images = len(image_files)
    step_size = max(1, num_images // 9)  # Ensure the interval is at least 1
    selected_images = [image_files[i] for i in range(0, num_images, step_size)]

    # Prepare subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('Test Set Inferences', fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])

            # Load image
            image = cv2.imread(image_path)

            # Check if the image is loaded correctly
            if image is not None:
                # Resize image
                resized_image = resize_image(image, size=(640, 640))
                # Normalize image
                normalized_image = normalize_image(resized_image)

                # Convert the normalized image to uint8 data type
                normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)

                # Predict with the model
                results = Valid_model.predict(source=normalized_image_uint8, imgsz=640, conf=0.5)

                # Plot image with labels
                annotated_image = results[0].plot(line_width=1)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ax.imshow(annotated_image_rgb)
            else:
                print(f"Failed to load image {image_path}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    ![Test set Images](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/download.png)
    **Step 17**
**Step-17 Deployment of app on Hugging face.**
    Go to https://huggingface.co/
    ![Hugging Face](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/Huggingface_logo.svg)
    Go to Spaces/
    Create new space/
**1-Add requirements.txt file**
    Add following libraries there\
            pandas\
            numpy\
            gradio \
            ultralytics\
            YOLO\
            ray==2.32.0\
            reportlab\
            Pillow==9.5.0\
            Ultralytics\
            joblib\
            reportlab\
            torchvision\
            torch\
            opencv-python-headless==4.8.0.74\
            
**2- Add app.py file**
    To create user interface in gradio write following script in app.py file\
    import os\
os.system("git lfs pull")\
import os\
os.system('pip install pillow')\
from PIL import Image\
# Your code here
import ultralytics\
print(f"Ultralytics version: {ultralytics.__version__}")\
from ultralytics import YOLO\
import logging\
import datetime\
import gradio as gr\
import numpy as np\
import torch\
import torchvision\
import cv2\
from PIL import Image, ImageDraw\
import joblib\
from reportlab.lib.pagesizes import letter\
from reportlab.pdfgen import canvas\

# Load the model ('best.pt' )
try:\
    model = YOLO('best.pt')\
except FileNotFoundError:\
    print("Model file 'best.pt' not found. Please upload it.")\
except Exception as e:\
    print(f"Error loading the model: {e}")\

def create_pdf_report(Date,name, gender, age, result, image_path, pdf_path):\
    """Creates a PDF report with the given result and patient particulars."""\
    c = canvas.Canvas(pdf_path, pagesize=letter)\
    width, height = letter\

    # Add title\
    c.setFont("Helvetica-Bold", 16)\
    c.drawString(100, height - 100, "Brain Tumor Detection Report")\

    # Add patient details and prediction result\
    c.setFont("Helvetica", 12)\
    c.drawString(100, height - 130, f"Date of Report: {Date}")\
    c.drawString(100, height - 150, f"Patient Name: {name}")\
    c.drawString(100, height - 170, f"Gender: {gender}")\
    c.drawString(100, height - 190, f"Age: {age}")\
    c.drawString(100, height - 210, f"Result: {result}")\

    # Add the MRI image\
    c.drawImage(image_path, 130, height - 550, width=300, height=300)\
     # Get current date\
    

    c.save()\

def predict_brain_tumor(image, name, gender, age):\
    """Predicts brain tumor presence in an image and draws bounding boxes."""\
    if model is None:\
        return "Model loading failed. Please check the model file.", None\

    try:\
        # Convert image to numpy array if required\
        image_array = np.array(image)\

        # Perform the prediction\
        prediction = model(image_array)  # Use YOLO's prediction method\
        logging.debug(f"Prediction output: {prediction}")\

        # Extract results and draw bounding boxes\
        result = "No brain tumor detected"\
        if len(prediction[0].boxes) > 0:  # Check if there are any bounding boxes\
            result = "Brain tumor detected"\
            annotated_image = np.array(image)  # Start with original image\

            for box in prediction[0].boxes\
                # Extract bounding box coordinates\
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates\
                conf = box.conf[0]  # Confidence score\
                label = f"{prediction[0].names[0]} ({conf:.2f})"\

                # Draw bounding box on image\
                annotated_image = cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)\
                annotated_image = cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,\
                                              0.5, (255, 0, 0), 2)\

            # Save the annotated image\
            image_path = "/tmp/annotated_image.png"\
            Image.fromarray(annotated_image).save(image_path)\
        else:\
            # Save the input image (unaltered)\
            image_path = "/tmp/input_image.png"\
            image.save(image_path)\

        # Save the Date\
        Date = datetime.date.today().strftime("%Y-%m-%d")\

        # Create the PDF report\
        pdf_path = "/tmp/Brain_Tumor_Detection_Report.pdf"\
        create_pdf_report(Date, name, gender, age, result, image_path, pdf_path)\

        return f"Date of Report: {Date}, Patient Name: {name}, Gender: {gender}, Age: {age}. Result: {result}", pdf_path\

    except Exception as e:\
        logging.error(f"Prediction error: {e}")\
        return f"Prediction error: {e}", None\
# Create the Gradio interface\
iface = gr.Interface(\
    fn=predict_brain_tumor,\
    inputs=[\
        gr.Image(type="pil"),\
        gr.Textbox(label="Patient Name"),\
        gr.Radio(choices=["Male", "Female", "Other"], label="Gender"),\
        gr.Number(label="Age")\
    ],\
    outputs=["text", "file"],\
    title="Brain Tumor Detection Report",\
    description="Upload MRI scan to detect brain tumors.",\
)\
iface.launch(share=True) \
Launch the interface\
3-step upload model
Go to add file> upload file> upload the mode"best.pt 
Now go to app will build and restart and take few moments. 
The interface would be like this 
![image](https://github.com/DS8-Trainee/Brain-Tumor-Cells-Detection/blob/main/image.png)\
*Methd of Use*
            The model has been deployed on huggingface space.\
            Go to (https://huggingface.co/stmuntahaa)\
            [stmuntahaa/Brain_Tumor_Detection](https://huggingface.co/stmuntahaa).\
            
            *1*-Deployed app's UI asks user to upload MRI scan and patient particulars (Patient name, Gender, Age), and prompts to click 
            the Submit button.
            *2*-The app produces results on UI (Output 0) along with a pdf report, that is downloadable. The report contains the                       necessary information on the patient's particulars and the result of the MRI scan.\
            The pdf report is downloaded to local Download folder of your device.
