#################################################
# Import all required modules
#################################################
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

#################################################
# Function_Name: GetData()
# Description: Function used to store data from a CSV file in a variable
# Input: CSV file path
# Output: DataFrame containing the data
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def GetData(Path):
    Data = pd.read_csv(Path)
    print("Top 5 entries of data: ", Data.head(5))
    return Data

#################################################
# Function_Name: Data_types()
# Description: Displays data types and info of the DataFrame
# Input: DataFrame
# Output: Data types and info printed to console
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def Data_types(Data):
    print(Data.dtypes)
    print("\n")
    print(Data.info())

#################################################
# Function_Name: DataInitialise()
# Description: Initializes feature and target variables
# Input: DataFrame
# Output: Feature and target variables
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def DataInitialise(Data):
    X = Data.drop('GradeClass', axis=1)
    Y = Data['GradeClass']  
    return X, Y

#################################################
# Function_Name: DataManipulation()
# Description: Splits data into training and testing sets
# Input: Feature and target variables
# Output: Training and testing datasets
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def DataManipulation(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

#################################################
# Function_Name: SortByGPA()
# Description: Sorts the data by GPA in descending order
# Input: DataFrame
# Output: Sorted DataFrame
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def SortByGPA(Data):
    sorted_data = Data.sort_values(by='GPA', ascending=False)
    print("Top 5 students by GPA:")
    print(sorted_data[['StudentID', 'GPA']].head(5))  # Assuming 'StudentID' is a column in your Data
    return sorted_data

#################################################
# Function_Name: Visualisation()
# Description: Visualizes data through various plots in a single figure
# Input: DataFrame
# Output: Plots displayed to console
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def Visualisation(Data):
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # Distribution of GradeClass
    sns.countplot(x='GradeClass', data=Data, palette='Set2', ax=axs[0, 0])
    axs[0, 0].set_title('Distribution of GradeClass')
    axs[0, 0].set_xlabel('Grade Class')
    axs[0, 0].set_ylabel('Count')

    # Study Time vs GPA Scatter Plot
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', hue='GradeClass', data=Data, palette='Set1', ax=axs[0, 1])
    axs[0, 1].set_title('Study Time vs GPA')
    axs[0, 1].set_xlabel('Study Time (Weekly)')
    axs[0, 1].set_ylabel('GPA')

    # Participation in Extracurricular Activities by Grade Class
    sns.countplot(x='Extracurricular', hue='GradeClass', data=Data, palette='pastel', ax=axs[1, 0])
    axs[1, 0].set_title('Participation in Extracurricular Activities by Grade Class')
    axs[1, 0].set_xlabel('Participates in Extracurricular')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend(title='Grade Class')

    # Parental Support vs GradeClass
    sns.countplot(x='ParentalSupport', hue='GradeClass', data=Data, palette='coolwarm', ax=axs[1, 1])
    axs[1, 1].set_title('Parental Support and Grade Class')
    axs[1, 1].set_xlabel('Parental Support')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend(title='Grade Class')

    # Age Histogram
    axs[2, 0].hist(Data["Age"], bins=8, color="green")
    axs[2, 0].set_title("Distribution of Age")
    axs[2, 0].set_xlabel("Age")
    axs[2, 0].set_ylabel("Count")

    # Gender Histogram
    axs[2, 1].hist(Data["Gender"], bins=2, color="purple")
    axs[2, 1].set_title("Gender Distribution")
    axs[2, 1].set_xlabel("Gender")
    axs[2, 1].set_ylabel("Count")

    # Adjust layout
    plt.tight_layout()
    plt.show()

#################################################
# Function_Name: DecisionTree_Classifier()
# Description: Implements Decision Tree classifier and prints accuracy
# Input: Training and testing datasets
# Output: Accuracy of the Decision Tree model
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    Dt = DecisionTreeClassifier(random_state=66)
    Dt.fit(X_train, Y_train)
    Y_pred = Dt.predict(X_test)
    print("Accuracy using DecisionTreeClassifier Algorithm: {:.2f}%".format(accuracy_score(Y_test, Y_pred) * 100))

#################################################
# Function_Name: RandomForest_Classifier()
# Description: Implements Random Forest classifier and prints accuracy
# Input: Training and testing datasets
# Output: Accuracy of the Random Forest model
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    RF = RandomForestClassifier(n_estimators=100)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm: {:.2f}%".format(accuracy_score(Y_test, Y_pred) * 100))

#################################################
# Function_Name: main()
# Description: Entry point function to call other functions
# Date: 29/07/2024
# Author: Rushikesh Waghmare
# Data: Student performance data from "Student_performance_data.csv"
#################################################
def main():
    Data = GetData("Student_performance_data.csv")
    Data_types(Data)
    X, Y = DataInitialise(Data)
    X_train, X_test, Y_train, Y_test = DataManipulation(X, Y)

    # Sort students by GPA
    sorted_data = SortByGPA(Data)

    # Visualization of data
    Visualisation(Data)
    
    # Model evaluations
    DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    RandomForest_Classifier(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()
