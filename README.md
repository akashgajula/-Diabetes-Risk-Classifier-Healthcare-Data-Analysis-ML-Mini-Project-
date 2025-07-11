# 🩺 Diabetes Risk Classifier – Healthcare Data Analysis (ML Mini-Project)

This project uses real-world patient data to build a diabetes risk prediction model using machine learning. The model is built using Python and evaluates medical indicators to help early diagnosis of diabetes.

---

## 📌 Overview

- 🎯 **Goal**: Predict the risk of diabetes based on health data.
- 📊 **Dataset**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 💻 **Tools Used**: Python, pandas, seaborn, matplotlib, scikit-learn

---

## 🧹 Data Cleaning

'''python
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

📊 Exploratory Data Analysis (EDA)

sns.countplot(x='Outcome', data=df)
plt.title('Outcome Distribution')
plt.show()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

⚙️ Feature Scaling & Splitting

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

🤖 Model Training & Evaluation

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

✅ Achieved 75% accuracy on test data 

---
## 🚀 Workflow

1. *Data Loading & Cleaning*
   - Handled invalid values (0s) for Glucose, BloodPressure, BMI, etc.
   - Replaced with mean values

2. *Exploratory Data Analysis (EDA)*
   - Distribution plots
   - Correlation heatmaps

3. *Feature Scaling & Splitting*
   - Used StandardScaler and train_test_split

4. *Model Training*
   - Applied *Logistic Regression*
   - Trained on 80% data and tested on 20%

5. *Model Evaluation*
   - Accuracy: 75% 
   - Confusion matrix & classification report

---

## 📈 Results

- The model predicted diabetes with around *75% accuracy*
- Showed balanced precision and recall for both classes (diabetic & non-diabetic)

---

## 📂 Files in the Repository

| File Name                                                              | Description                               |
|------------------------------------------------------------------------|-------------------------------------------|
| Healthcare Data Analysis (Mini-Project) – Diabetes Prediction.ipynb    | Complete notebook with code and outputs   |
| README.md                                                              | This file                                 |

---

📚 Learnings
Preprocessing healthcare data for machine learning
Training and evaluating classification models
Understanding model metrics in a medical context

👨‍💻 Author
Akash Gajula
📧 akashgajula1@gmail.com
[LinkedIn](https://www.linkedin.com/in/akash-gajula-a2a4a22a6)
