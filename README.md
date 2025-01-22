# neural-network-image-classifier
Neural network for image classification using PyTorch

This program is an image classification system using deep learning with PyTorch. It takes a set of images (in this case, simple shapes like circles, squares, and triangles) and trains a neural network to recognize and classify these shapes. The program consists of three main parts:

1. **Data Preparation**: It reads images from a folder and their corresponding labels from a CSV file. Each image is associated with a label (0 for circles, 1 for squares, 2 for triangles) so the network knows what it's supposed to learn.

2. **Neural Network Model**: The program uses a convolutional neural network (CNN) called ModernNet that's designed to process images. This network learns to identify patterns and features in the images that help distinguish between different shapes.

3. **Training Process**: During training, the program shows each image to the network, compares the network's prediction with the correct label, and adjusts the network's parameters to improve its accuracy. The training progresses through multiple epochs (complete passes through all images), and the program regularly reports the loss (error rate) and accuracy to show how well the network is learning.

The end goal is to create a model that can accurately classify new images of shapes it hasn't seen before. The React component I added provides a visual interface to generate and display sample images that can be used for training.


Create_dataset.py generates a dataset consisting of 30 samples  
With the following variations:  
-Random colors (5 different colors)  
-Random sizes (between 60 and 100 pixels)  
-Random positions on the image  
-Rotation for squares  
-Shape distortion for triangles  
  
Equal distribution of shapes (10 of each type)  
  
Circles are labeled as 0  
Squares are labeled as 1  
Triangles are labeled as 2  


<h2>SAMPLE OUTPUT</h2>

![Image](https://github.com/user-attachments/assets/e130e012-05e8-453a-b9d4-e9de9c8415bc)  
![Image](https://github.com/user-attachments/assets/3bda4306-5ea3-44cd-b7fd-e7108f5ad375)



