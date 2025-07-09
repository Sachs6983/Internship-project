# Internship-project
Student Performance Predictor for EduQuest Coaching
Overview
This project is a machine learning model designed to predict a student's final exam score based on various features such as gender, age, parental education, family income, internet access, previous exam scores, attendance rate, homework completion rate, class participation score, number of absences, extra-curricular involvement, learning hours per week, and tutor support.
Features

Data loading and exploration
Data preparation
Training multiple machine learning models (Linear Regression, Random Forest, Gradient Boosting, Support Vector Regression)
Selecting the best model based on R² score
Hyperparameter tuning for the best model
Feature importance analysis
Interactive function to predict student performance

Technologies Used

Python
pandas
numpy
scikit-learn
matplotlib
seaborn

Installation
To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
scikit-learn
matplotlib
seaborn

You can install these libraries using pip:
pip install pandas numpy scikit-learn matplotlib seaborn

Usage

Ensure you have the input CSV file. The default file is "Student Performance Predictor for EduQuest Coaching.csv". You can specify a different file by passing the filename to the StudentPerformancePredictor constructor.
Run the script. It will perform the complete analysis, including training models and selecting the best one.
After the analysis is complete, you can use the predict_student_performance function to make predictions for new students. You will be prompted to enter the student's details, and the model will predict the final exam score.

Limitations

The model assumes that the input data is in a specific format and that the features are relevant to predicting the final exam score.
The predicted final exam score is capped between 0 and 100.

License
This project is licensed under the MIT License.
