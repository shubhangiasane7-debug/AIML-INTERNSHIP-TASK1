import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("archive/Titanic-Dataset.csv")

# Show basic info
print(df.info())
print(df.isnull().sum())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Encode categorical column
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Standardize numerical column
scaler = StandardScaler()
df[['Age']] = scaler.fit_transform(df[['Age']])

# Boxplot for outliers
sns.boxplot(x=df['Age'])
plt.show()

print("Preprocessing Completed")
