import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample Dataset (Previous Scores, Study Hours, Class Participation)
data = {
    'Previous_Score': [75, 88, 60, 55, 95, 45, 70, 50, 85, 80],
    'Study_Hours': [5, 8, 3, 2, 10, 1, 4, 2, 7, 6],
    'Class_Participation': [8, 9, 5, 4, 10, 2, 6, 3, 9, 7],
    'Exam_Score': [78, 90, 62, 58, 98, 50, 72, 55, 89, 83]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Previous_Score', 'Study_Hours', 'Class_Participation']]
y = df['Exam_Score']

# Model Training
model = LinearRegression()
model.fit(X, y)

# Save Model (Pickle)
with open('exam_score_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully as exam_score_model.pkl")
