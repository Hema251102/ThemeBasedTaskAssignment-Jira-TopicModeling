# Theme-Based Task Assignment using Topic Modeling on Jira

This repository provides the code and instructions required to implement theme-based task assignment using topic modeling on Jira issue tracking data. The methodology applies machine learning techniques to automate the assignment of tasks based on their themes, optimizing resource allocation and improving workflow efficiency.

## Overview

The project explores how topic modeling can be used to assign tasks more efficiently in a Jira issue-tracking system by categorizing issues into different themes. Using Natural Language Processing (NLP) and machine learning, it automatically suggests appropriate assignees for tasks based on these themes.

## Instructions for Running the Code

Follow these steps to reproduce the methodology and results described in this repository.

### 1. Retrieving the Data

Before running any code, download the dataset required for this project:

- **Download Dataset:** [Click here to download the dataset](https://doi.org/10.5281/zenodo.5665895)

Once downloaded, configure the properties file (`properties.py`) to set up your environment and data paths.

### 2. Running the Methodology

The core of this project involves several steps, which are executed through Python scripts. Follow the steps below to process the data and run the topic modeling methodology.

1. **Retrieve the Data:**
   - Run the script `1_mongo_get_and_save_data.py` to retrieve the data from the database.
   
2. **Preprocess the Features:**
   - Execute the script `2_features_preprocess_and_transform.py` to preprocess and transform the features into a usable format for analysis.

3. **Text Preprocessing:**
   - Run `3_text_preprocessing.py` to clean and preprocess the text data, including tokenization and removal of stop words.

4. **Prepare the Train and Test Sets:**
   - Run `4_prepare_train_test_sets.py` to split the dataset into training and testing sets for model evaluation.

5. **Apply Classification:**
   - Execute `5_apply_classification.py` to apply classification algorithms on the prepared data.

6. **Optimize Topics:**
   - Finally, run `6_optimize_topics.py` to fine-tune the topic models and obtain the best results.

After running these steps, the results will be saved in the folder specified in the `properties.py` file. The output files are saved as zip files and follow a consistent naming format (e.g., `3_{project_name}_{num_assignees}_assignees.csv` for step `3_text_preprocessing.py`).

### 3. Generating Tables and Figures

Once the models are trained and results are obtained, you can reproduce the tables and graphs shown in the research by running the following evaluation scripts:

- **AUC Evaluation:**
  - `7_1_evaluation_auc.py` – Generates AUC (Area Under the Curve) metrics for model evaluation.
  
- **Evaluation by Number of Assignees:**
  - `7_2_evaluation_num_assignees.py` – Analyzes results based on the number of assignees per task.
  
- **Evaluation by Assignees:**
  - `7_3_evaluation_assignees.py` – Provides insights into the performance across different assignees.

- **Label Evaluation:**
  - `7_4_evaluation_labels.py` – Evaluates model performance based on the assigned labels.

- **Classifier Performance Evaluation:**
  - `7_5_evaluation_classifiers.py` – Compares the performance of various classifiers used in the modeling process.

### Results and Output

Each of the above scripts will generate output files and figures, which will be saved in the data folder you configured in `properties.py`. These results will include classification metrics, visualizations, and model performance reports, providing insights into the task assignment process and the effectiveness of topic modeling.

---

## Dependencies

Before running the code, make sure you have the following libraries installed:

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`
- `gensim`
- `joblib`

You can install them using pip:

```bash
pip install -r requirements.txt
