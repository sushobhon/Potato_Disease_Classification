# Potato_Disease_Classification
An Image Classification Project to identify Potato Disease Classification

# The Data Source
The Dataset is downloaded from Kaggle using this link: https://www.kaggle.com/datasets/arjuntejaswi/plant-village
The dataset also has images of other plants.

# How to Run?
To Use the model first clone the githun repo in your local using below command
```
git clone https://github.com/sushobhon/Potato_Disease_Classification.git
```

Then Change the directory
```
cd Potato_Disease_Classification
```
Install all the requirments
```
pip install -r requirements.txt
```
One model is saved in saved models folder. `PotatoImgClassification.py` is used to build the model.

# Calling The Model in Web Page
First run `main.py` in `api` folder to start backend. Once Backend is running run Frontend.

There is a Frontend HTML code present in the `api` folder. To use Frontend run "Forntend.html" in your web browser. Upload a potato leave image. The api will call the model and prediction will be visiable in the page.
