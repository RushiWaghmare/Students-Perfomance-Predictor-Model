# Students-Perfomance-Predictor-Model

## Overview
This project aims to predict student performance based on various academic and demographic factors using machine learning techniques. The analysis utilizes a dataset stored in an Excel file and applies several algorithms to uncover insights that could support educational strategies.

## Dataset
The dataset is provided in an Excel (.csv) file named `Student_performance_data.csv` and contains the following features:

- **Demographics**:
  - Age
  - Gender
  - Ethnicity
  - Parental Education

- **Academic Factors**:
  - Study Time Weekly
  - Absences
  - GPA
  - Grade Class

- **Extracurriculars**:
  - Sports
  - Music
  - Volunteering
  - Parental Support

Ensure the Excel file is placed in the project folder, and the script will automatically load it for analysis.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the Excel file is named `Student_performance_data.csv` and placed in the project folder.
2. Run the main script to preprocess the data, train the model, and evaluate its performance:
   ```bash
   python main.py
   ```

## Code Structure
The code consists of several functions organized to facilitate data loading, preprocessing, visualization, and model evaluation:

- **GetData(Path)**: Loads data from a specified CSV file.
- **Data_types(Data)**: Displays data types and information of the DataFrame.
- **DataInitialise(Data)**: Initializes feature and target variables.
- **DataManipulation(X, Y)**: Splits the data into training and testing sets.
- **SortByGPA(Data)**: Sorts the data by GPA and prints the top students.
- **Visualisation(Data)**: Creates various visualizations for data exploration.
- **DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)**: Implements a Decision Tree classifier and prints accuracy.
- **RandomForest_Classifier(X_train, X_test, Y_train, Y_test)**: Implements a Random Forest classifier and prints accuracy.

## Algorithms Used
This project employs the following machine learning algorithms:
- **Decision Tree Classifier**: Achieved an accuracy of **93.04%**.
- **Random Forest Classifier**: Achieved an accuracy of **91.64%**.

## Performance Metrics
The model's performance is evaluated based on accuracy:
- **Decision Tree Accuracy**: 93.04%
- **Random Forest Accuracy**: 91.64%

## Conclusion
This project demonstrates how machine learning can be applied to predict student performance, providing insights that could help educators tailor their approaches to meet students' individual needs. The findings highlight the importance of various factors in academic success.
