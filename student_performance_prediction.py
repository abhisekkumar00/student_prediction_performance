
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate synthetic data
data = {
    'study_hours': np.random.randint(1, 10, 100),
    'attendance': np.random.randint(60, 100, 100),
    'previous_scores': np.random.randint(40, 100, 100),
}

# Create pass/fail based on simple logic
df = pd.DataFrame(data)
df['pass'] = np.where((df['study_hours'] >= 5) & (df['attendance'] >= 75) & (df['previous_scores'] >= 50), 1, 0)

# 2. Exploratory Data Analysis
sns.pairplot(df, hue='pass')
plt.show()

# 3. Split data
X = df[['study_hours', 'attendance', 'previous_scores']]
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict on new input
def predict_result(study_hours, attendance, previous_scores):
    input_data = np.array([[study_hours, attendance, previous_scores]])
    prediction = model.predict(input_data)
    return "Pass" if prediction[0] == 1 else "Fail"

# Example usage
print(predict_result(6, 85, 70))  # Output: Pass or Fail
