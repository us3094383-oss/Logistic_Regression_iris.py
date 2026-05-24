# Logistic_Regression_iris.py
# Logistic Regression on Iris Dataset 

# Project Structure
Imports: scikit-learn modules for dataset, model, and evaluation
Dataset: Iris dataset(`load_iris`)
Model: Logistic Regression (`LogisticRegression`)
Evaluation: Accuracy Score & Log Loss

# Workflow
1. Load Dataset:  
   Features: Sepal length, Sepal width, Petal length, Petal width  
   Target: Iris species (Setosa, Versicolor, Virginica)

2. Split Dataset:  
    Training: 80%  
    Testing: 20%  
    Random state fixed for reproducibility

3. Train Model:
   Logistic Regression with(`max_iter=200`)

4. Predict & Evaluate
   Predictions (`predict`)  
   Class probabilities (`predict_proba`)  
   Accuracy (`accuracy_score`)  
   Loss (`log_loss`)

# Results
  Predicted Labels: First 10 predictions displayed  
  Class Probabilities: First 10 probability distributions displayed  
  Final Metrics:
  Accuracy: ~0.9-1(varies slightly depending on split)  
  Log Loss: ~0.1–0.2  
