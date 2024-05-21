Report on the Neural Network Model for Alphabet Soup
Overview of the Analysis
The purpose of this analysis is to build and evaluate a deep learning model to predict the success of funding applications for Alphabet Soup, a hypothetical charity organization. The goal is to develop a model that can accurately classify whether an application will be successful based on various features of the application.

Results
Data Preprocessing
Target Variable:
The target variable for the model is the success of the funding application.
Feature Variables:
The features for the model include various attributes of the funding application, such as application type, applicant information, funding amount, etc.
Variables to Remove:
Any variables that are neither targets nor features, such as unique identifiers (e.g., application ID), were removed from the input data to prevent them from influencing the model.
Compiling, Training, and Evaluating the Model
AlphabetSoupCharity1:

Model Architecture:
File: Deep_Learning_Challenge1
Layers: 3
Neurons:
First layer: 8 units, activation: ReLU, input_dim: 43
Second layer: 5 units, activation: ReLU
Output layer: 1 unit, activation: Sigmoid
Training Parameters: 50 epochs
Results:
Accuracy: 72.55%
Loss: 0.5581
Rationale:
The choice of 8 units in the first layer was based on initial experimentation. ReLU was selected as the activation function for the hidden layers due to its effectiveness in deep learning models. The output layer uses a sigmoid activation to provide a probability score for the binary classification.
AlphabetSoupCharity2:

Model Architecture:
File: Deep_Learning_Challenge2
Layers: Variable
Neurons:
First layer: 1-10 units, activation: ReLU or Tanh (decided by Keras Tuner)
Hidden layers: 1-10 units, activation: ReLU or Tanh (decided by Keras Tuner)
Output layer: 1 unit, activation: Sigmoid
Training Parameters: 80 epochs, validation_split: 0.2
Results:
Accuracy: 71.65%
Loss: 0.5724
Rationale:
Keras Tuner was used to optimize the model by selecting the best activation function and number of units in the first and hidden layers. This approach allows the model to dynamically adjust to the most effective architecture for the given data.
AlphabetSoupCharity3:

Model Architecture:
File: Deep_Learning_Challenge2
Layers: Variable
Neurons:
First layer: 1-8 units, activation: ReLU or Tanh (decided by Keras Tuner)
Hidden layers: 1-8 units, activation: ReLU or Tanh (decided by Keras Tuner)
Output layer: 1 unit, activation: Sigmoid
Training Parameters: 100 epochs
Results:
Accuracy: 70.79%
Loss: 0.5851
Rationale:
This model further refined the hyperparameters by limiting the units to 1-8 in each layer and increasing the training epochs to 100. The goal was to fine-tune the model for better performance.
Summary
Model Performance:
AlphabetSoupCharity1: Best accuracy (72.55%) with a loss of 0.5581.
AlphabetSoupCharity2: Slightly lower accuracy (71.65%) with a loss of 0.5724.
AlphabetSoupCharity3: Lowest accuracy (70.79%) with a loss of 0.5851.
Recommendation:
Despite the variations in accuracy and loss, the first model (AlphabetSoupCharity1) performed the best. However, the performance metrics indicate that none of the models achieved a high enough accuracy to be deemed reliable for predicting funding success. Further optimization, including exploring different model architectures, feature engineering, and advanced techniques like ensemble learning, may be necessary to improve model performance.
