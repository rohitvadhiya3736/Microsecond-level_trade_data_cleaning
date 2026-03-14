import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   FRAUD DETECTION - COMPLETE PYTHON ANALYSIS")
print("=" * 60)

print("\nSTEP 1: Loading Data...")

df = pd.read_csv('/mnt/user-data/uploads/trades.csv')
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")
print(f"   Columns : {list(df.columns)}")

print("\nSTEP 2: Data Exploration...")
print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe().round(2))

print("\n--- Label Distribution ---")
print(df['label'].value_counts())
print(f"   Fraud Rate: {df['label'].mean()*100:.2f}%")

print("\n--- Payment Methods ---")
print(df['paymentMethod'].value_counts())

print("\nSTEP 3: Data Cleaning...")

# Rename columns
df = df.rename(columns={
    'label'     : 'Fraud_Flag',
    'localTime' : 'Transaction_Time'
})

# Fix missing values
df = df.assign(
    accountAgeDays       = df['accountAgeDays'].fillna(df['accountAgeDays'].median()),
    numItems             = df['numItems'].fillna(0),
    paymentMethodAgeDays = df['paymentMethodAgeDays'].fillna(df['paymentMethodAgeDays'].median()),
    paymentMethod        = df['paymentMethod'].fillna('unknown')
)

# Fix data types
df['accountAgeDays']       = df['accountAgeDays'].astype(int)
df['numItems']             = df['numItems'].astype(int)
df['paymentMethodAgeDays'] = df['paymentMethodAgeDays'].round(2)
df['paymentMethod']        = df['paymentMethod'].str.lower().str.strip()

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"   Duplicates removed : {before - len(df)}")

# Remove outliers
df = df[df['accountAgeDays'] <= 2000]
df = df[df['numItems'] <= df['numItems'].quantile(0.99)]
df = df[df['paymentMethodAgeDays'] >= 0]
df = df.reset_index(drop=True)
print(f"   Clean rows         : {len(df)}")
