# Pizza vs. Steak Classification with TensorFlow CNN

## Project Overview

This project involves building a Convolutional Neural Network (CNN) using TensorFlow to classify images of pizza and steak. The goal is to develop a robust image classification model that can accurately distinguish between these two food items, leveraging the power of deep learning and CNNs for visual recognition tasks.

## Dataset

- **Source:** The dataset contains labeled images of pizza and steak.
- **Classes:** Two classes - `Pizza` and `Steak`.

## Tools & Libraries Used

- **Data Handling:**
  - `TensorFlow` and `Keras` for building and training the CNN model.
  - `Pandas` for handling data preprocessing (if applicable).
- **Image Processing:**
  - `OpenCV` or `PIL` for image loading and preprocessing.
- **Model Evaluation:**
  - Metrics like accuracy, precision, recall, and confusion matrix to evaluate the model’s performance.

## Methodology

### Data Preprocessing:

- **Image Resizing:**
  - Resized all images to a consistent size to ensure uniformity in the input data.
  
- **Data Augmentation:**
  - Applied techniques like rotation, flipping, and zooming to increase the diversity of the training data and reduce overfitting.

### Model Development:

- **CNN Architecture:**
  - Constructed a Convolutional Neural Network with multiple convolutional and pooling layers to extract features from images.
  - Used activation functions like ReLU and softmax for non-linearity and classification.
  
- **Model Training:**
  - Trained the CNN using a categorical cross-entropy loss function and the Adam optimizer.
  - Implemented early stopping and checkpointing to prevent overfitting and save the best model.

### Model Evaluation:

- **Accuracy and Loss:**
  - Monitored accuracy and loss curves during training to assess model convergence.
  
- **Confusion Matrix:**
  - Analyzed the confusion matrix to evaluate model performance on each class.

- **Example Usage:**
  ```python
  model.predict(new_image)
  ```

## Results

The CNN model effectively classified images of pizza and steak, achieving high accuracy. The model was able to generalize well to unseen images, demonstrating the power of CNNs in image classification tasks.

## Conclusion

This project showcases the application of Convolutional Neural Networks for binary image classification. The model’s success indicates the potential for using deep learning in various visual recognition tasks.

## Future Work

- Expand the dataset to include more food categories, improving the model’s robustness and generalization.
- Experiment with different CNN architectures and hyperparameters to further enhance performance.
- Deploy the model as a web application or mobile app for real-time food recognition.
