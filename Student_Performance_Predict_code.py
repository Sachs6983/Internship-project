import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class StudentPerformancePredictor:
    def __init__(self, csv_file="Student Performance Predictor for EduQuest Coaching.csv"):
        self.csv_file = csv_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_pipeline = None
        self.preprocessor = None
        self.model_results = {}
        
    def load_and_explore_data(self):
        """Load data and perform initial exploration"""
        print("=" * 60)
        print("STUDENT PERFORMANCE PREDICTOR - EduQuest Coaching")
        print("=" * 60)
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Check for missing values
            print("\nMissing values in each column:")
            missing_values = self.df.isnull().sum()
            print(missing_values)
            
            # Basic statistics
            print("\nBasic Statistics:")
            print(self.df.describe())
            
            # Check target variable distribution
            print(f"\nTarget variable (final_exam_score) statistics:")
            print(f"Mean: {self.df['final_exam_score'].mean():.2f}")
            print(f"Std: {self.df['final_exam_score'].std():.2f}")
            print(f"Min: {self.df['final_exam_score'].min():.2f}")
            print(f"Max: {self.df['final_exam_score'].max():.2f}")
            
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found!")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
        return True
    
    def prepare_data(self):
        """Prepare data for modeling"""
        # Define feature columns
        categorical_cols = ['gender', 'parental_education', 'internet_access', 
                          'extra_curricular_involvement', 'tutor_support']
        numerical_cols = ['age', 'family_income', 'previous_exam_score', 'attendance_rate', 
                         'homework_completion_rate', 'class_participation_score', 
                         'number_of_absences', 'learning_hours_per_week']
        
        # Separate features and target
        X = self.df.drop('final_exam_score', axis=1)
        y = self.df['final_exam_score']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numerical_cols)
            ]
        )
        
        print("\nData preparation completed!")
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def train_multiple_models(self):
        """Train and compare multiple ML models"""
        print("\n" + "=" * 50)
        print("TRAINING MULTIPLE MODELS")
        print("=" * 50)
        
        # Define models to compare (Ridge and Lasso removed)
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            # Fit the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            self.model_results[name] = {
                'pipeline': pipeline,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"  MSE: {mse:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.3f}")
            print(f"  CV R² (mean ± std): {cv_mean:.3f} ± {cv_std:.3f}")
    
    def select_best_model(self):
        """Select the best model based on R² score"""
        print("\n" + "=" * 50)
        print("MODEL COMPARISON RESULTS")
        print("=" * 50)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.model_results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': results['r2'],
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'CV R² Mean': results['cv_mean'],
                'CV R² Std': results['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = best_model_name
        self.best_pipeline = self.model_results[best_model_name]['pipeline']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best R² Score: {comparison_df.iloc[0]['R² Score']:.3f}")
        
        return comparison_df
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print(f"\n" + "=" * 50)
        print(f"HYPERPARAMETER TUNING FOR {self.best_model}")
        print("=" * 50)
        
        if self.best_model == 'Random Forest':
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
        elif self.best_model == 'Gradient Boosting':
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__min_samples_split': [2, 5, 10]
            }
        elif self.best_model == 'Support Vector Regression':
            param_grid = {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'model__kernel': ['rbf', 'linear', 'poly']
            }
        else:
            print("No hyperparameter tuning implemented for this model.")
            return
        
        # Create base pipeline
        if self.best_model == 'Random Forest':
            base_model = RandomForestRegressor(random_state=42)
        elif self.best_model == 'Gradient Boosting':
            base_model = GradientBoostingRegressor(random_state=42)
        elif self.best_model == 'Support Vector Regression':
            base_model = SVR()
        
        pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', base_model)
        ])
        
        # Perform grid search
        print("Performing Grid Search...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Update best pipeline
        self.best_pipeline = grid_search.best_estimator_
        
        # Evaluate tuned model
        y_pred_tuned = self.best_pipeline.predict(self.X_test)
        r2_tuned = r2_score(self.y_test, y_pred_tuned)
        rmse_tuned = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(f"Test R² score (tuned): {r2_tuned:.3f}")
        print(f"Test RMSE (tuned): {rmse_tuned:.3f}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Categorical features (one-hot encoded)
        categorical_cols = ['gender', 'parental_education', 'internet_access', 
                          'extra_curricular_involvement', 'tutor_support']
        
        # Get feature names from the preprocessor
        try:
            cat_encoder = self.best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        except:
            print("Could not extract categorical feature names")
        
        # Numerical features
        numerical_cols = ['age', 'family_income', 'previous_exam_score', 'attendance_rate', 
                         'homework_completion_rate', 'class_participation_score', 
                         'number_of_absences', 'learning_hours_per_week']
        feature_names.extend(numerical_cols)
        
        # Calculate permutation importance
        try:
            perm_importance = permutation_importance(
                self.best_pipeline, self.X_test, self.y_test, n_repeats=10, random_state=42
            )
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(perm_importance.importances_mean)],
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False, float_format='%.4f'))
            
        except Exception as e:
            print(f"Could not calculate feature importance: {e}")
    
    def predict_student_performance(self):
        """Interactive function to predict student performance"""
        print("\n" + "=" * 60)
        print("STUDENT PERFORMANCE PREDICTION")
        print("=" * 60)
        
        print("Enter the following student details:")
        
        try:
            # Get student details
            gender = input("Gender (Male/Female): ").strip()
            age = int(input("Age: "))
            parental_education = input("Parental Education (High School/Undergraduate/Graduate): ").strip()
            family_income = float(input("Family Income: "))
            internet_access = input("Internet Access (Yes/No): ").strip()
            previous_exam_score = float(input("Previous Exam Score: "))
            attendance_rate = float(input("Attendance Rate (%): "))
            homework_completion_rate = float(input("Homework Completion Rate (%): "))
            class_participation_score = float(input("Class Participation Score: "))
            number_of_absences = int(input("Number of Absences: "))
            extra_curricular_involvement = input("Extra Curricular Involvement (Low/Moderate/High): ").strip()
            learning_hours_per_week = float(input("Learning Hours per Week: "))
            tutor_support = input("Tutor Support (Yes/No): ").strip()
            
            # Create DataFrame for prediction
            student_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'parental_education': [parental_education],
                'family_income': [family_income],
                'internet_access': [internet_access],
                'previous_exam_score': [previous_exam_score],
                'attendance_rate': [attendance_rate],
                'homework_completion_rate': [homework_completion_rate],
                'class_participation_score': [class_participation_score],
                'number_of_absences': [number_of_absences],
                'extra_curricular_involvement': [extra_curricular_involvement],
                'learning_hours_per_week': [learning_hours_per_week],
                'tutor_support': [tutor_support]
            })
            
            # Make prediction and clip to [0, 100]
            predicted_score = np.clip(self.best_pipeline.predict(student_data)[0], 0, 100)
            
            print(f"\n" + "=" * 40)
            print(f"PREDICTION RESULT")
            print(f"=" * 40)
            print(f"Predicted Final Exam Score: {predicted_score:.2f}")
            
            # Provide performance category
            if predicted_score >= 90:
                category = "Excellent"
            elif predicted_score >= 80:
                category = "Good"
            elif predicted_score >= 70:
                category = "Average"
            elif predicted_score >= 60:
                category = "Below Average"
            else:
                category = "Poor"
            
            print(f"Performance Category: {category}")
            
            # Provide recommendations
            print(f"\nRecommendations:")
            if predicted_score < 70:
                print("- Consider additional tutoring support")
                print("- Increase study hours and homework completion rate")
                print("- Improve attendance and class participation")
            elif predicted_score < 80:
                print("- Maintain current study habits")
                print("- Focus on areas of weakness")
                print("- Consider joining study groups")
            else:
                print("- Keep up the excellent work!")
                print("- Consider mentoring other students")
                print("- Explore advanced learning opportunities")
            
        except ValueError:
            print("Invalid input! Please enter numeric values where required.")
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        if not self.load_and_explore_data():
            return
        
        self.prepare_data()
        self.train_multiple_models()
        self.select_best_model()
        self.hyperparameter_tuning()
        self.analyze_feature_importance()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("The model is now ready for predictions.")
        print("You can now use predict_student_performance() to make predictions.")

# Example usage
if __name__ == "__main__":
    # Initialize the predictor with the default dataset
    predictor = StudentPerformancePredictor()
    
    # Run complete analysis
    predictor.run_complete_analysis()
    
    # Make predictions for new students
    while True:
        print("\n" + "=" * 60)
        choice = input("Do you want to predict performance for a new student? (y/n): ").strip().lower()
        if choice == 'y':
            predictor.predict_student_performance()
        else:
            break
    
    print("\nThank you for using the Student Performance Predictor!")
    print("EduQuest Coaching - Empowering Students for Success!")