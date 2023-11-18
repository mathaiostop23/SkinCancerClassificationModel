# Skin Cancer Classification Model Using CNN


 ## Overview 

 This project focuses on skin cancer classification using a deep learning model implemented in Keras. 
 The model is trained on the HAM10000 dataset, which consists of dermatoscopic images of skin lesions categorized into seven different classes. 
 The goal is to accurately classify these lesions into their respective types, such as melanocytic nevi,
 melanoma, benign keratosis-like lesions, basal cell carcinoma, actinic keratoses, vascular lesions, and dermatofibroma.

 ### Dependencies

- Python 3.10
- Tensorflow Keras 2.13
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- PIL

## Dataset

The HAM10000 dataset is utilized for training and testing the model. 
This dataset contains 10015 diverse images of multi-source dermatoscopic images of pigmented lesions, each labeled with its corresponding type. 
The dataset is preprocessed, and each image is resized to (100 x 75) pixels to fit the model's input requirements.

## Data Processing

- Size of Training Set: 8012
- Size of Validation Set: 1001
- Size of Test Set: 1002
- Shape of Images: (75, 100, 3)
- Number of Unique Classes: 7

**Data Augmentation:**

Data augmentation is performed using Keras's ImageDataGenerator to introduce variations in the training set, 
including rotation, zooming, shifting, and flipping. This helps the model generalize better to unseen data.

**Normalization:**

Pixel values of images are normalized to ensure convergence during training and improve accuracy. 
The mean and standard deviation of the training set are used for normalization.

**Label Encoding:**

Labels are one-hot encoded using Keras's to_categorical to match the output format required for categorical crossentropy loss.


Samples of each Class : 

![category_samples](https://github.com/mathaiostop23/SkinCancerClassificationModel/assets/75705991/9408c2a9-2df3-4993-b251-621e2ebf1537)


## Model Architecture

**Convolutional Neural Network (CNN):**

-The model consists of multiple convolutional layers that capture hierarchical features from input images.
-Batch Normalization is applied after convolutional layers for improved stability during training.

**MaxPooling:**

-Max pooling layers are incorporated to downsample spatial dimensions and focus on essential features.

**Dropout:**

-Dropout layers (e.g., Dropout(0.5)) are included to prevent overfitting by randomly setting a fraction of input units to zero during training.

**Dense Layers:**

-Fully connected dense layers interpret the extracted features and make predictions.
-The final layer has softmax activation for multi-class classification.



## Training

**Hyperparameters :**

- Optimizer : Adam
- Learning Rate : 0.001
- Epsilon : 1e-05
- Beta1 : 0.9
- Beta2 : 0.999
- Epochs : 40
- Batch Size : 32


**Results :**

- Training Accuracy : 78.41%
- Validation Accuracy : 72.53%
- Test Accuracy : 70.16%

## Summary 

The model is designed for skin cancer classification, leveraging data preprocessing, a CNN architecture, and data augmentation. 
After training for 40 epochs, the model achieves competitive accuracy on the test set. 
The provided example script can be used for further exploration, adaptation, and contribution to enhance skin cancer classification.

## Last Details

The model was trained on a Macbook Pro 2021 with M1 Pro on a Conda environment.

## References 

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/code?datasetId=54339&sortBy=voteCount


