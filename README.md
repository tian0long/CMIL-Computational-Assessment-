#Binary Classification of globally sclerotic and non-sclerotic glomeruli

Introduction: Glomeruli, crucial components of the kidney, play an essential role in filtering blood to maintain healthy renal function. Structural abnormalities in glomeruli are often early indicators of renal diseases. Therefore, accurately distinguishing between globally sclerotic and non-sclerotic glomeruli is critical for effective diagnosis and treatment of kidney-related disorders.

In this project, I implemented a deep learning-based approach using the ResNet18 architecture, pretrained on the ImageNet dataset, to classify glomeruli images. To ensure consistency and facilitate effective training, various data preprocessing techniques such as Contrast Limited Adaptive Histogram Equalization (CLAHE), sample rebalance, image resizing and normalization were employed. The pretrained ResNet18 model was adjusted for binary classification to distinguish between non-sclerotic and sclerotic glomeruli. Experimental results demonstrated that the model effectively identifies the two categories with high accuracy, precision, and recall. 

## ML Pipeline:

The Machine Learning pipline includes the following steps:

1. Library Imports and Setup: Essential Python libraries such as Pandas, PyTorch, cv2 and Matplotlib are imported. Setup involves configuring paths for data storage (`data/`) and results output (`results/`), and initializing hyperparameters like learning rate and batch size.

2. Data Reading and Description: The annotations from the `public.csv` file are read using Pandas. This CSV file contains image annotations which are crucial for understanding the data distribution and guiding further processing for model training.

3.  Data Visualization: 

   Category Distribution: The balance between sclerotic and non-sclerotic samples is visualized using a bar plot, addressing any potential class imbalance in the dataset.
   

4. Dataset Handling and Preparation: Creation of a `GlomeruliDataset` class, inheriting from `torch.utils.data.Dataset`, is defined for efficient data loading and preprocessing. This includes image pre-processing with CLAHE, sample rebalance, image resizing, normalization, and tensor conversion. The dataset is split into training, validation, and testing sets.
5. Model Configuration: The ResNet18 model, pre-trained on ImageNet, is adapted for the binary classification task. Unfreeze the last 4 layers for transfer learning and the last fully connected layer is modified to output two classes.
6.  Model Training: The training process is carried out over a specified number of epochs. During training, model parameters are updated using the Adam optimizer, and learning rate adjustments are made with a StepLR scheduler. Training and validation losses and accuracies are tracked for performance monitoring.
7.  Model Saving: Post-training, the model's state dictionary is saved for future evaluation and deployment, ensuring that the trained model can be reused without the need for retraining.
8.  Model Evaluation: The model is evaluated on a separate testing set using metrics like accuracy, precision, recall, and F1 score. Additional performance insights are provided through a confusion matrix, precision-recall curve, and ROC curve.

## Dataset and Preprocessing
The dataset is organized into two sub-folders within the `data/` directory: `globally_sclerotic_glomeruli` and `non_globally_sclerotic_glomeruli`, containing image patches for sclerotic and non-sclerotic classes, respectively. The `public.csv` file lists the image patches with their corresponding labels. Since the samples is not balance between those two categories, we go through the data augmentation for sclerotic, which increase the number of samples to 4 times. Then use CLAHE to improve contract, avoid over-amplification of noise. 

The dataset consists of 8920 histopathological images labeled as globally sclerotic (1) or non-globally sclerotic (0). Each image underwent preprocessing, including resizing, normalization, and augmentation, to ensure consistent input to the neural network.

## Training and Validation
ResNet18 as a residual learning framework for deep convolutional neural networks, is selected for this task due to its efficiency in learning from a relatively small amount of data and its ability to generalize well on image recognition tasks. Pretrained on ImageNet, it has proven to be an effective architecture for transfer learning, especially in medical imaging.

I adopted the ResNet18 model, modifying the final layer to output binary classifications. The dataset was split into training (60%), validation (20%), and test sets (20%). 

## Performance Metrics

The model's performance is evaluated based on accuracy, precision, recall, and F1 score. These metrics are chosen to provide a comprehensive understanding of the model's ability to classify sclerotic and non-sclerotic glomeruli accurately.
-  Accuracy:  99.55%, indicating a high level of overall correct predictions.
-  Precision: 99.88%, showing the model's accuracy when predicting positive classes.
-  Recall:  99.17%, highlighting the model's ability to detect most positive instances.
-  F1 Score:  99.52%, balancing precision and recall, suggests a robust model performance across both classes.

## Conclusion

The high performance of the [model](https://www.dropbox.com/scl/fo/ip7pxm8zgm4t23qes0e15/h?rlkey=23u1hed6lla5kbzcuc2e9muna&dl=0) suggests that transfer learning and fine-tuning of pre-trained networks are effective for binary classification in histopathological images.

## Environment Setup

Ensure you have Conda installed. Clone the project repository, then create and activate the project environment using:

   ```
   conda env create -f environment.yml
   conda activate project_env
   ```

## Model Training and Test

Execute the following command to train and test the model:

   ```
   python train_test.py
   ```

## Model Evaluation
 Model download:  [model](https://mstate-my.sharepoint.com/:u:/g/personal/lt766_msstate_edu/EW9JAetFYkVMvtKanT3OpKABIR8vISNylRlNZosZXO4Egg?e=iC8E9z)

After training the model, you can evaluate it on a new set of images by using the `evaluation.py` script. The script requires the path to a folder containing glomeruli image patches as input and outputs a CSV file with the model's predictions.

Run the script as follows:

   ```
   python evaluation.py 
   ```

Add  image folder path, model.pth path and evaluation.csv path in line 112,113,114.

## Dependencies

This project is developed in PyCharm, under an Anaconda environment, and utilizes PyTorch for the deep learning components. Ensure you have Anaconda and PyCharm installed on your system. The following libraries and tools are required:

-  Python: Ensure you have Python installed, ideally through Anaconda. This project was developed using Python 3.10.
-  PyTorch: Used for all deep learning operations, including neural network definition and training. 
-  Pandas: For data manipulation and analysis. 
-  cv2: For CLAHE.
-  Pillow (PIL): For image processing tasks.
-  Matplotlib:  For creating static, interactive, and animated visualizations in Python.
-  Seaborn:  For making statistical graphics in Python. It is built on top of Matplotlib. 
-  Scikit-learn: For machine learning utilities such as data splitting. 
-  Warnings: Typically included with the standard Python library, used to suppress warnings.

For a complete list of dependencies, refer to the `environment.yml` file.

## Project Files
- `train.py`: Script for training the model.
- `evaluation.py`: Script for evaluating the model on a new set of glomeruli image patches.
- `data/`: Folder containing the dataset and `public.csv` with image names and their corresponding labels.
- `model/`: Directory where the trained model is saved.
- `environment.yml`: Conda environment file to set up the required environment for running the project.
- `test/`: Contains test data. 

# Reference
Ayyar, Meghna, Puneet Mathur, Rajiv Ratn Shah, and Shree G. Sharma. 
"Harnessing ai for kidney glomeruli classification." 
In 2018 IEEE International Symposium on Multimedia (ISM), pp. 17-20. IEEE, 2018. 
https://doi.org/10.1109/ISM.2018.00011
