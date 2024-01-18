# Sugarcane Leaf Disease Detection using a CNN

Discover the capabilities of this Convolutional Neural Network (CNN) meticulously trained on a diverse dataset comprising over 2000 images. Designed to address agricultural challenges, the model excels at identifying and classifying four distinct types of sugarcane leaf diseases: RedRot, Yellow, Mosaic and Rust. This project leverages advanced deep learning techniques to provide an effective solution for automated disease detection in sugarcane crops.

## Convolution Algorithm (Computer Vision - Convolution Neural Networks)

### Input Image

The process begins with an input image, which is a matrix of pixel values. In color images, there are typically three channels (Red, Green, and Blue).

### Convolution Opeartion

Convolution is the core operation in CNNs. It involves applying a filter (also known as a kernel or feature detector) to the input image. The filter is a smaller matrix than the input image and slides over the image with a specified stride.

### Element-Wise Multiplication

At each position, the filter is overlaid on a region of the input image, and element-wise multiplication is performed between the filter and the corresponding region of the input.

### Summation

The results of the element-wise multiplication are summed to produce a single value. This operation is repeated as the filter slides across the entire image.

### Feature Map

The result of the convolution operation is a feature map. It highlights specific patterns or features that the filter is designed to detect.

### Activation Function

Typically, a non-linear activation function, such as ReLU (Rectified Linear Unit), is applied element-wise to the feature map. This introduces non-linearity to the model, allowing it to learn complex patterns.

### Pooling (Subsampling)

Pooling layers (e.g., Max Pooling or Average Pooling) are often used to reduce the spatial dimensions of the feature maps, making the model more robust and computationally efficient. Pooling involves selecting the maximum or average value from a group of neighboring pixels.

### Flattening

The output from convolutional and pooling layers is flattened into a one-dimensional vector. This vector serves as the input for fully connected layers.

### Fully Connected Layers

The flattened vector is connected to one or more fully connected layers, which perform classification tasks. These layers use weights to combine features from the previous layers.

### Output Layer

The final fully connected layer produces an output that represents the class probabilities for the input image. The model is trained using a loss function, and optimization algorithms adjust the weights during training to minimize the loss.

### Softmax - Our Activation Function

In classification tasks, a softmax activation function is commonly used in the output layer to convert the raw scores into class probabilities.

This process is repeated during both the forward pass (making predictions) and the backward pass (backpropagation for training) in the neural network. The convolutional layers learn to detect hierarchical features, starting from simple patterns to complex structures, enabling CNNs to excel in image-related tasks.

![59954intro to CNN](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/e028a5e2-c9ab-4224-987e-075e0bb4ecda)

## Dataset Overview

A dataset of sugarcane leaf diseases was meticulously compiled through manual collection. It encompasses five primary categories: Healthy, Mosaic, Redrot, Rust, and Yellow diseases. The dataset, comprising a total of 2569 images across these categories, was captured using smartphones with diverse configurations to ensure variability. Originating from Maharashtra, India, the database maintains balance and exhibits a rich variety. Notably, image sizes are not uniform, stemming from the diverse capturing devices. All images are encoded in the RGB format.

### Healthy

Example of a healthy lead:

![healthy (419)](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/ab11ef9c-f4a2-447c-82e2-a7fea881cec6)

### Mosaic Disease

Example of a leaf with mosaic disease:

![mosaic (371)](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/d7283a41-98ce-4456-b763-f2b65abaafaa)

### Redrot Disease

Example of a leaf with redrot disease:

![redrot (416)](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/93784cac-146c-4342-982b-877734d2a521)

### Rust Disease

Example of a leaf with rust disease:

![rust (413)](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/28d6d6e3-7263-47b7-8c9b-52b43b46361a)

### Yellow Disease

Example of a leaf with yellow disease:

![yellow (405)](https://github.com/Neill-Erasmus/sugarcane-leaf-disease-detection/assets/141222943/ca5f2dae-9fa3-42f2-b3b9-2ab2579f03a9)

## Architecture of the Convolutional Neural Network

The Convolutional Neural Network (CNN) architecture employed in this project is designed for sugarcane leaf disease classification. The model consists of multiple layers, starting with convolutional layers responsible for learning hierarchical features from the input images. These convolutional layers are followed by MaxPooling layers to downsample the spatial dimensions of the feature maps, reducing computational complexity. The output is then flattened and connected to fully connected layers, facilitating high-level feature extraction and classification. Rectified Linear Unit (ReLU) activation functions introduce non-linearity throughout the network, aiding in capturing complex patterns. The final layer utilizes the softmax activation function to produce class probabilities. The model is trained using a dataset containing images of sugarcane leaves affected by various diseases, allowing it to learn and distinguish between Healthy, Mosaic, RedRot, Rust, and Yellow diseases. Regularization techniques, such as dropout, are employed to prevent overfitting. Overall, this CNN architecture is designed to effectively classify sugarcane leaf diseases with robust feature learning and generalization capabilities.

## Training and Evaluation of the Neural Network

The training of the Convolutional Neural Network (CNN) spanned 25 epochs, each consisting of batches processed through the dataset. During the initial epoch, the model showed a relatively low accuracy of 26.65% on the training set and 28.69% on the validation set. However, as training progressed, the accuracy improved significantly, reaching 85.04% on the training set and 73.71% on the validation set by the final epoch. The loss function, measuring the disparity between predicted and actual values, decreased over epochs, indicating learning convergence.

```
Epoch 1/25
64/64 [==============================] - 18s 275ms/step - loss: 1.5482 - accuracy: 0.2665 - val_loss: 1.6973 - val_accuracy: 0.2869
Epoch 2/25
64/64 [==============================] - 18s 284ms/step - loss: 1.3027 - accuracy: 0.4235 - val_loss: 1.5470 - val_accuracy: 0.4382
Epoch 3/25
64/64 [==============================] - 19s 298ms/step - loss: 1.1410 - accuracy: 0.5255 - val_loss: 1.3662 - val_accuracy: 0.4960
Epoch 4/25
64/64 [==============================] - 40s 629ms/step - loss: 1.1012 - accuracy: 0.5527 - val_loss: 1.0704 - val_accuracy: 0.6036
Epoch 5/25
64/64 [==============================] - 44s 696ms/step - loss: 0.9760 - accuracy: 0.6127 - val_loss: 1.1715 - val_accuracy: 0.6255
Epoch 6/25
64/64 [==============================] - 44s 696ms/step - loss: 0.9265 - accuracy: 0.6439 - val_loss: 1.1004 - val_accuracy: 0.6394
Epoch 7/25
64/64 [==============================] - 16s 249ms/step - loss: 0.8415 - accuracy: 0.6766 - val_loss: 1.1140 - val_accuracy: 0.6434
Epoch 8/25
64/64 [==============================] - 16s 248ms/step - loss: 0.7990 - accuracy: 0.7048 - val_loss: 1.1693 - val_accuracy: 0.6474
Epoch 9/25
64/64 [==============================] - 18s 276ms/step - loss: 0.8017 - accuracy: 0.6964 - val_loss: 0.9397 - val_accuracy: 0.7052
Epoch 10/25
64/64 [==============================] - 18s 286ms/step - loss: 0.7378 - accuracy: 0.7058 - val_loss: 0.9043 - val_accuracy: 0.7052
Epoch 11/25
64/64 [==============================] - 21s 323ms/step - loss: 0.7498 - accuracy: 0.7246 - val_loss: 0.9375 - val_accuracy: 0.6733
Epoch 12/25
64/64 [==============================] - 21s 336ms/step - loss: 0.6552 - accuracy: 0.7499 - val_loss: 1.1179 - val_accuracy: 0.6295
Epoch 13/25
64/64 [==============================] - 18s 288ms/step - loss: 0.6661 - accuracy: 0.7434 - val_loss: 1.0790 - val_accuracy: 0.6912
Epoch 14/25
64/64 [==============================] - 19s 298ms/step - loss: 0.6139 - accuracy: 0.7687 - val_loss: 0.9144 - val_accuracy: 0.7550
Epoch 15/25
64/64 [==============================] - 19s 293ms/step - loss: 0.5952 - accuracy: 0.7642 - val_loss: 1.1632 - val_accuracy: 0.6375
Epoch 16/25
64/64 [==============================] - 19s 297ms/step - loss: 0.5850 - accuracy: 0.7761 - val_loss: 0.9580 - val_accuracy: 0.7430
Epoch 17/25
64/64 [==============================] - 30s 478ms/step - loss: 0.5461 - accuracy: 0.7905 - val_loss: 1.2235 - val_accuracy: 0.7052
Epoch 18/25
64/64 [==============================] - 17s 259ms/step - loss: 0.5300 - accuracy: 0.8039 - val_loss: 1.1416 - val_accuracy: 0.7072
Epoch 19/25
64/64 [==============================] - 20s 305ms/step - loss: 0.5406 - accuracy: 0.7855 - val_loss: 1.2130 - val_accuracy: 0.6912
Epoch 20/25
64/64 [==============================] - 19s 297ms/step - loss: 0.4979 - accuracy: 0.8187 - val_loss: 0.8970 - val_accuracy: 0.7689
Epoch 21/25
64/64 [==============================] - 20s 308ms/step - loss: 0.4837 - accuracy: 0.8172 - val_loss: 1.2952 - val_accuracy: 0.6892
Epoch 22/25
64/64 [==============================] - 20s 314ms/step - loss: 0.4327 - accuracy: 0.8410 - val_loss: 1.0595 - val_accuracy: 0.7351
Epoch 23/25
64/64 [==============================] - 29s 461ms/step - loss: 0.4230 - accuracy: 0.8430 - val_loss: 1.2714 - val_accuracy: 0.6952
Epoch 24/25
64/64 [==============================] - 20s 314ms/step - loss: 0.3934 - accuracy: 0.8504 - val_loss: 0.8988 - val_accuracy: 0.7470
Epoch 25/25
64/64 [==============================] - 27s 419ms/step - loss: 0.4222 - accuracy: 0.8504 - val_loss: 0.9571 - val_accuracy: 0.7371
```

### Overfitting

Overfitting occurs when a machine learning model learns the training data too well, capturing noise and specific details that are unique to the training set but may not generalize well to new, unseen data. In essence, the model becomes too tailored to the idiosyncrasies of the training data, resulting in poor performance on validation or test datasets.

A lower model accuracy on the training set is often preferred to avoid overfitting. This might seem counterintuitive, but it signifies that the model is not memorizing the training data but rather learning the underlying patterns and features that are more likely to generalize to new data. A model with slightly lower training accuracy but better generalization capability is preferred because it is more likely to perform well on unseen data, demonstrating its ability to make accurate predictions in real-world scenarios beyond the training set. Regularization techniques, such as dropout and weight regularization, are commonly employed to help prevent overfitting and promote better generalization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
