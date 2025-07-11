import pandas as pd
# Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
#Data load
#path = "/content/data.csv"
#df=pd.read_csv(path)
df=pd.read_csv("/content/data.csv")
df.head(5)
#df.to_string()

#Check shape and basic info
df.shape
df.info()
df.describe()

#Check for missing values
df.isnull().sum()

#Check for duplicate or irrelevant columns
df.duplicated().sum()
df.drop_duplicates(inplace=True)

#Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
plt.show()

#Data Preprocessing & Feature Engineering
#Encode categorical columns
df = pd.get_dummies(df, drop_first=True)
print(df)

#Train-Test Split
from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop('Calories', axis=1)
y = df['Calories']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

#Train a Model & Evaluate
#RandomForestClassifier (Regression)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data.csv')

# Handle missing values
df['Calories'] = df['Calories'].fillna(df['Calories'].median())

# Feature engineering
df['Pulse_Diff'] = df['Maxpulse'] - df['Pulse']
df['Intensity_Ratio'] = df['Pulse'] / df['Maxpulse']

# Prepare data
X = df.drop('Calories', axis=1)
y = df['Calories']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
