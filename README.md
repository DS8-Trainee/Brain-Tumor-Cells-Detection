# Brain-Tumor-Cells-Detection
This project will detect tumor in brain MRI scans.
The dataset has been taken from Roboflow (having one class :tumor)
It uses Yolov8n model
All major libraries and packages have been listed in the dependencies file.
Model has been given epoch 16, batch size=-1, optimiser=auto
The model has been checked on unseen data from Hayatabad Hospital, Peshawar. It is giving fairly good results.
After training the model, it has been saved and Gradio-based UI was made.
The model has been deployed on huggingface space "stmuntahaa/Brain_Tumor_Detection".
Deployed app's UI asks user to upload MRI scan and patient particulars (Patient name, Gender, Age), and prompts to click the Submit button.
The app produces results on UI (Output 0) along with a pdf report, that is downloadable. The report contains the necessary information on the patient's particulars and the result of the MRI scan. The pdf report is downloaded to local Download folder of your device.
