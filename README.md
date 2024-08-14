# Phytoplankton Classification - README

## Project Overview

This project focuses on the classification of phytoplankton species using two datasets: FlowCam and Imaging FlowCytobot (IFCB). The datasets contain images and, in the case of FlowCam, also geometric features. We preprocess these datasets and use them to train and evaluate four machine learning models: Support Vector Classifier (SVC), Multi-Layer Perceptron (MLP), ResNet (with and without image augmentation), and a Combined or Fusion Model. This README provides detailed instructions on how to set up the environment, preprocess the data, train the models, and evaluate the results.

## Prerequisites

Before you begin, ensure you have Python installed. You can install the required Python packages using the `requirements.txt` file provided.

### Installation Instructions

1. **Set up the Python environment**:
   Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized into several folders and notebooks as described below:

## Notebooks

1. **Data Preprocessing**:
   - `step_1_data_preprocessing_flowcam.ipynb`: Preprocesses the FlowCam data in three stages.
   - `step_2_data_preprocessing_IFCB.ipynb`: Preprocesses the IFCB data in three stages.
   - `step_3_combining_data.ipynb`: Combines processed FlowCam and IFCB data.

2. **Model Training**:
   - `step_4_svm_and_mlp_model.ipynb`: Defines, loads, and trains the SVC and MLP models.
   - `step_5_resnet_model_and_fusion_model.ipynb`: Defines, loads, and trains the ResNet models (with and without image augmentation) and the Fusion model.

3. **Model Evaluation**:
   - `step_6_model_evaluation.ipynb`: Loads saved model weights, generates classification results, and confusion matrices.
   - `step_7_data_comparison_flowcam_IFCB.ipynb`: Compares processed data from FlowCam and IFCB datasets and finds common categories.


### Folders

- `raw_flowcam_data/`: Contains the raw FlowCam dataset.
- `raw_ifcb_data/`: Contains the raw IFCB dataset.

- `flowcam_processed_1/`: Stores the results of "step-1_data_preprocessing_flowcam.ipynb" which processess the initial FlowCam data.
- `flowcam_merged_data/`: Stores the processed data after merging FlowCam images and features from `flowcam_processed_1/` .
- `flowcam_split_data/`: Stores the final split (training, validation, test) FlowCam data after processing data from `flowcam_merged_data/`.

- `ifcb_processed_1/`: Stores the results of "step-2_data_preprocessing_IFCB.ipynb" which processess the initial FlowCam data.
- `ifcb_merged_data/`: Stores the processed data after merging IFCB images and image extracted features from `ifcb_processed_1/`.
- `ifcb_split_data/`: Stores the final split (training, validation, test) IFCB data after processing data from `ifcb_merged_data/`.

- `common_processed_data/`: Stores the results of "step-3_combining_data.ipynb" which process the combined data from FlowCam and IFCB datasets.
- `common_merged_data/`: Stores the processed data after merging FlowCam and IFCB datasets from `common_processed_data/`.
- `common_split_data/`: Stores the final split (training, validation, test) combined data from `common_merged_data/`.


- `results/`: Contains the classification results.
- `trained_models/`: Stores the trained model weights.
- `actual_trained_model_weights/`: Pre-saved best performing model weights inside "trained_models" folder.

### Utility Files

- `utils.py`: Contains utility functions used across notebooks.
- `data_preprocessing.py`: Contains functions for data preprocessing steps.


## Step-by-Step Instructions

### 1. Data Preprocessing

Start by preprocessing the raw datasets. The notebooks should be executed in the following order:

1. **FlowCam Data Preprocessing**:
   - Open and run `step_1_data_preprocessing_flowcam.ipynb`.
   - This notebook will process the data in three stages and store the results in the following folders:
     - `flowcam_processed_1/`
     - `flowcam_merged_data/`
     - `flowcam_split_data/`


2. **IFCB Data Preprocessing**:
   - Open and run `step_2_data_preprocessing_IFCB.ipynb`.
   - This notebook will process the IFCB data in three stages and store the results in the following folders:
     - `ifcb_processed_1/`
     - `ifcb_merged_data/`
     - `ifcb_split_data/`

3. **Combining Data**:
   - Open and run `step_3_combining_data.ipynb`.
   - This notebook combines the processed data from FlowCam and IFCB datasets and stores the results in the following folders:
     - `common_processed_data/`
     - `common_merged_data/`
     - `common_split_data/`

At the end of this step, the final split data (training, validation, test) for FlowCam, IFCB, and combined datasets will be available in their respective folders.

### 2. Model Training

With the processed data ready, you can now train the machine learning models:

1. **SVC and MLP Models**:
   - Open and run `step_4_svm_and_mlp_model.ipynb`.
   - This notebook will define, load, and train the SVC and MLP models. The classification results will be saved in the `results/` folder, and the trained model weights will be stored in the `trained_models/` folder.

2. **ResNet and Fusion Models**:
   - Open and run `step_5_resnet_model_and_fusion_model.ipynb`.
   - This notebook will define, load, and train the ResNet models (with and without image augmentation) and the Fusion model. The classification results will be saved in the `results/` folder, and the trained model weights will be stored in the `trained_models/` folder.

**Note:** Training deep learning models, particularly ResNet and Fusion models, requires significant computational power. Pre-trained best performance model weights are already provided in the `actual_trained_model_weights/` folder.
a. mlp_flowcam_trained_model.h5  
b. resnet_with_aug_flowcam_trained_model.h5           
c. resnet_with_aug_ifcb_trained_model.h5
d. fusion_combined_data_trained_model.h5

### 3. Model Evaluation

After training the models, evaluate their performance:

1. **Model Evaluation**:
   - Open and run `step_6_model_evaluation.ipynb`.
   - This notebook loads the saved model weights and generates classification results and corresponding confusion matrices. It compares the performance of different models across datasets.
2. **Data Comparison**:
   - Open and run `step_7_data_comparison_flowcam_IFCB.ipynb`.
   - This notebook compares the first stage processed data from the FlowCam and IFCB datasets and finds common categories. The results are visualized in bar graphs.

## Additional Notes

- High Computational Resources: For training deep learning models, it is recommended to use a machine with high computational power or access to GPU resources.
- Model Weights: Pre-trained model weights are provided for convenience. If you choose to retrain models, ensure your machine has sufficient resources.
- Confusion Matrices: Use confusion matrices generated in the evaluation step to analyze the categories with lower performance (F1 score < 0.5) and discuss possible improvements.

