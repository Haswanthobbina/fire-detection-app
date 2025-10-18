Forest Fire Detection with CNN and Streamlit
This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images and detect the presence of forest fires. The model is deployed in an interactive web application using Streamlit.
Overview
The goal of this project is to provide a user-friendly tool for fire detection. A user can upload an image, and the trained CNN model will analyze it and predict whether it contains a fire or not, displaying the result with a confidence score.
The sidebar of the application shows the model's performance metrics from its training and validation phases.
Technologies Used
Python
TensorFlow / Keras
Streamlit
Pillow
NumPy
Pandas
Google Colab (for model training)
Setup and Installation
Because the trained model file (FFD.keras) is too large for a standard GitHub repository, you will first need to generate it by running the included Google Colab notebook.
Part 1: Generating the Model (Google Colab)
Open the Notebook: Open the fire_detection_week3.ipynb file in Google Colab. You can use this direct link:
Run the Cells: In the Colab environment, run all the cells from top to bottom (Runtime > Run all). This will download the dataset, build the model, and train it.
Download the Model: The final cell in the notebook will save the trained model as FFD.keras. Download this file to your local machine.
Part 2: Running the Streamlit App (Locally)
Clone the Repository:
git clone [https://github.com/Haswanthobbina/fire-detection-app.git](https://github.com/Haswanthobbina/fire-detection-app.git)
cd fire-detection-app


Create a Virtual Environment:
python -m venv venv

Activate it:
Windows: .\venv\Scripts\activate
Mac/Linux: source venv/bin/activate
Install Dependencies:
pip install -r requirements.txt


Add the Model File: Place the FFD.keras file you downloaded from Colab into the main fire-detection-app directory.
Run the App:
streamlit run app.py

Your web browser will open with the application running locally!
Author
Obbina Haswanth * LinkedIn Profile 
