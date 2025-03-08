# Brain-Tumor-Cells-Detection\
This project will detect tumor in brain MRI scans to help assist the radiologist/expert in detecting tumor accurately and within less time.\
**1st Step:** The dataset has been taken from Roboflow (having one class:tumor)\
            Go to Roboflow Universe or follow the link\
            (Link:https://universe.roboflow.com/zaky-indra-w86c4/brain-tumor-gsh0d/dataset/1).
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
            ![Recall for model](




The model has been checked on unseen data from Hayatabad Hospital, Peshawar. It is giving fairly good results.
After training the model, it has been saved and Gradio-based UI was made.
The model has been deployed on huggingface space "stmuntahaa/Brain_Tumor_Detection".
Deployed app's UI asks user to upload MRI scan and patient particulars (Patient name, Gender, Age), and prompts to click the Submit button.
The app produces results on UI (Output 0) along with a pdf report, that is downloadable. The report contains the necessary information on the patient's particulars and the result of the MRI scan. The pdf report is downloaded to local Download folder of your device.
