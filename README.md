# importing packages
import numpy as np
import pandas as pd
# Reading dataset
df = pd.read_csv('your_dataset.csv')
df.head()
# See the no. of rows and columns
df.shape
#printing all columns
data_cols = df.columns.tolist()
print(data_cols)
# Display data type of each feature
df.dtypes
df.describe()
df.info()
# find the null values
df.isnull().sum()
# Remove unwanted columns
df.drop(columns=['column1', 'column2'], inplace=True) # specify the
columns to remove
# Fill numerical columns with mean and categorical columns with the
mode
num_columns = df.select_dtypes(include=['float64', 'int64']).columns
cat_columns = df.select_dtypes(include=['object']).columns
df[num_columns] = df[num_columns].fillna(df[num_columns].mean()) #
numerical columns
df[cat_columns] =
df[cat_columns].fillna(df[cat_columns].mode().iloc[0]) # categorical
columns
# fill the missing values for numerical terms - mean
#df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
# fill the missing values for categorical terms - mode
#df['Gender'] = df["Gender"].fillna(df['Gender'].mode()[0])
# find the null values
df.isnull().sum()
# Encode categorical features using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in cat_columns:
df[column] = label_encoder.fit_transform(df[column])
# Scale numerical features with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])
# Split data into train and test sets
# replace 'target_column' with your target column name
X = df.drop('target_column', axis=1) # -- independent variable
y = df['target_column'] # -- dependent variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
# Evaluate Logistic Regression model
from sklearn.metrics import accuracy_score, confusion_matrix
logreg_pred = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_conf_matrix = confusion_matrix(y_test, logreg_pred)
print("Logistic Regression Accuracy:", logreg_accuracy)
print("Logistic Regression Confusion Matrix:\n", logreg_conf_matrix)
#Train Random Forest Classifier model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# Evaluate Random Forest Classifier model
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Plotting the first tree in the forest
plot_tree(rf_model.estimators_[0], filled=True)
plt.show()
# Select the best model
best_model = logreg_model if logreg_accuracy > rf_accuracy else
rf_model
# Save the best model as a .pkl file
import pickle
with open('best_model.pkl', 'wb') as file:
pickle.dump(best_model, file)
print("Best model saved as 'best_model.pkl'")
