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
#################################################
def DataManipulation(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

#################################################
# Function_Name: SortByGPA()
# Description: Sorts the data by GPA in descending order
# Input: DataFrame
# Output: Sorted DataFrame
#################################################
def SortByGPA(Data):
    sorted_data = Data.sort_values(by='GPA', ascending=False)
    print("Top 5 students by GPA:")
    print(sorted_data[['StudentID', 'GPA']].head(5))  # Assuming 'StudentID' is a column in your Data
    return sorted_data

#################################################
# Function_Name: Visualisation()
# Description: Visualizes data through various plots
# Input: DataFrame
# Output: Plots displayed to console
#################################################
def Visualisation(Data):
    # Distribution of GradeClass
    plt.figure(figsize=(8, 6))
    sns.countplot(x='GradeClass', data=Data, palette='Set2')
    plt.title('Distribution of GradeClass')
    plt.xlabel('Grade Class')
    plt.ylabel('Count')
    plt.show()

    # Study Time vs GPA Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', hue='GradeClass', data=Data, palette='Set1')
    plt.title('Study Time vs GPA')
    plt.xlabel('Study Time (Weekly)')
    plt.ylabel('GPA')
    plt.show()

    # Participation in Extracurricular Activities by Grade Class
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Extracurricular', hue='GradeClass', data=Data, palette='pastel')
    plt.title('Participation in Extracurricular Activities by Grade Class')
    plt.xlabel('Participates in Extracurricular')
    plt.ylabel('Count')
    plt.legend(title='Grade Class')
    plt.show()

    # Parental Support vs GradeClass
    plt.figure(figsize=(8, 6))
    sns.countplot(x='ParentalSupport', hue='GradeClass', data=Data, palette='coolwarm')
    plt.title('Parental Support and Grade Class')
    plt.xlabel('Parental Support')
    plt.ylabel('Count')
    plt.legend(title='Grade Class')
    plt.show()

    # Age Histogram
    plt.hist(Data["Age"], bins=8, color="green")
    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

    # Gender Histogram
    plt.hist(Data["Gender"], bins=2, color="purple")
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()

#################################################
# Function_Name: DecisionTree_Classifier()
# Description: Implements Decision Tree classifier and prints accuracy
# Input: Training and testing datasets
# Output: Accuracy of the Decision Tree model
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
#################################################
def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    RF = RandomForestClassifier(n_estimators=100)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm: {:.2f}%".format(accuracy_score(Y_test, Y_pred) * 100))

#################################################
# Function_Name: main()
# Description: Entry point function to call other functions
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
