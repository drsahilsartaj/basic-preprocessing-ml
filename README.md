# basic-preprocessing-ml
## Creating Feature Matrix (X) and Target Variable (y)

```python
X = dataset.iloc[:, :-1].values  # Select all columns except the last one
y = dataset.iloc[:, -1].values   # Select only the last column as the target
```
----------------------------------

# 📌 Handling Missing Data in a Dataset

## ❓ When Do We Use This?
If a column contains **numerical missing values**, we can replace them with the **mean** of that column. This ensures the dataset remains consistent for machine learning models.

---

## ✅ Solution: Using `SimpleImputer` to Fill Missing Values
To automatically replace missing values with the **mean**, we use `SimpleImputer` from `sklearn.impute`:

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Create an imputer instance with the 'mean' strategy
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Apply imputer to numerical columns with missing values
imputer.fit(dataset[['Feature1', 'Feature2', 'Feature3']])

# Transform dataset by replacing missing values with the mean
dataset[['Feature1', 'Feature2', 'Feature3']] = imputer.transform(dataset[['Feature1', 'Feature2', 'Feature3']])

# 📌 Alternative Strategies: 'median', 'most_frequent', 'constant' can be used in place of 'mean' for different scenarios.
# (When missing values are in categorical columns (use mode instead).)
```
----------------------------------------------

# 📌 Handling Categorical Independent Variables

## ❓ When Do We Use This?
In datasets, some **independent variables (features)** are **categorical**, meaning they contain text values like:
- **Categories:** Red, Blue, Green  
- **Locations:** New York, Paris, London  
- **Job Titles:** Engineer, Doctor, Teacher  

Since machine learning models **only work with numbers**, we need to **convert categorical variables into numerical form** before training.

---

## ✅ Solution: Using `ColumnTransformer` and `OneHotEncoder`
To **convert categorical data** into numerical form, we use `OneHotEncoder` inside `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Apply OneHotEncoding to all categorical columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')

# Example:
# Apply OneHotEncoding to the first column (categorical variable) ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Transform the dataset and convert it into a NumPy array
X_transformed = np.array(ct.fit_transform(X))
```
----
## ❌ When Not to Use OneHotEncoding?
OneHotEncoding is **not always the best choice**. Here are cases where it might not be suitable:

1️⃣ **When categorical variables have too many unique values (High Cardinality)**  
   - Example: City names, ZIP codes, unique product IDs.  
   - Instead, use **Label Encoding** or **Target Encoding**.

2️⃣ **When categorical data has an order (Ordinal Data)**  
   - Example: **Education Level** (High School < Bachelor's < Master's < PhD).  
   - Instead, use **Ordinal Encoding**.

3️⃣ **When it creates too many columns (Sparse Matrix Problem)**  
   - If a dataset has a categorical column with **1000 unique categories**, OneHotEncoding will create **1000 new columns**!  
   - Instead, consider **Feature Hashing or Embeddings**.

-------------------------------------------------------
# 📌 Encoding the Dependent Variable

## ❓ When Do We Use This?
In machine learning, the **dependent variable (target)** is what we are predicting.  
Sometimes, the dependent variable is **categorical** (e.g., Yes/No, Spam/Not Spam, Disease/No Disease), but models only understand numbers.  
So, we need to **convert categorical labels into numeric values** using `LabelEncoder`.

---

## ✅ Solution: Using `LabelEncoder` for Encoding the Dependent Variable
To **convert categorical target values** into numerical form, we use `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
le = LabelEncoder()

# Fit and transform the dependent variable
y_encoded = le.fit_transform(y)
---------------------------------------------------------------
# 📌 Splitting the Dataset into Training and Test Sets

## ❓ When Do We Use This?
Before training a machine learning model, we need to **split the dataset** into:
- **Training Set** → Used to train the model.
- **Test Set** → Used to evaluate the model’s performance.

This helps in assessing how well the model generalizes to unseen data.

---

## ✅ Solution: Using `train_test_split`
To split the dataset into training and test sets, we use `train_test_split` from `sklearn.model_selection`:

```python
from sklearn.model_selection import train_test_split

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #X: Matrix of features and y: Dependent variable vector the o/p
```
---
## ❌ When Not to Use a Simple Train-Test Split?

### 1️⃣ When Working with Small Datasets
- If the dataset is too small, using **only one train-test split** may not give reliable results.
- Instead, use **cross-validation** to train on different subsets multiple times.

### 2️⃣ When Data is Time-Dependent (Time Series)
- If the dataset involves **time-sensitive data** (e.g., stock prices, weather forecasting), a **random split can break the order**.
- Instead, use **`TimeSeriesSplit`** to split data sequentially.

### 3️⃣ When the Dataset is Imbalanced
- If one class (e.g., **"Spam" or "Not Spam"**) is much larger than the other, a random split may **not preserve the ratio**.
- Instead, use **`StratifiedShuffleSplit`** to ensure **equal representation of all classes** in training and test sets.

---------------------------------------------
# 📌 Feature Scaling in Machine Learning

## ❓ Why Use Feature Scaling?
Feature scaling makes sure all values are in a similar range so that:
- No feature dominates others.
- Models learn faster and work better.
- Distance-based models give correct results.

---
## ✅ When to Use Feature Scaling?

### 1️⃣ When Using Distance-Based Models
```python
from sklearn.preprocessing import StandardScaler  # Import the scaler

sc = StandardScaler()  # Create a scaler object
X_train = sc.fit_transform(X_train)  # Fit on training data and scale it
X_test = sc.transform(X_test)  # Use the same scaling on test data
```

```
      StandardScaler() scales data so that it has mean = 0 and standard deviation = 1.
      .fit_transform(X_train) learns scaling from training data and applies it.
      .transform(X-test) uses the same scaling on test data to keep consistency.
```

### 2️⃣ When Using Distance-Based Models
```python
# Models like Logistic Regression and Neural Networks train faster with scaling.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
## ❌ When Not to Use Feature Scaling?
### 1️⃣ When Using Tree-Based Models
```# Decision Trees, Random Forests, and XGBoost do not need scaling.```
### 2️⃣ When Features Are Already Similar
```# If all values are already in a small range (e.g., 0-1), scaling is not needed.```
### 3️⃣ When Working with Categorical Data
```# Do not scale text-based or labeled data. Use One-Hot Encoding instead.```

------------------------------
# 📌 .fit_transform() vs .transform()
```
✅ What is .fit_transform()?

Think of it like a chef tasting food before adding salt.

X_train = sc.fit_transform(X_train)

	•	fit() → Learns how much salt (mean & standard deviation) is needed.
	•	transform() → Uses that recipe to scale the data.
	•	The model remembers this scaling for future use.

✅ What is .transform()?

Now, imagine the chef already knows how much salt to use. They just apply it to new food.

X_test = sc.transform(X_test)

	•	Does not learn anything new.
	•	Only applies the same scaling learned from training data.
	•	Keeps test data in the same scale as training data.

❌ Why Not Use .fit_transform() on Test Data?

X_test = sc.fit_transform(X_test)  # ❌ WRONG!

	•	If you learn new scaling on test data, it’s like the chef changing the salt recipe for every new dish.
	•	This is called data leakage, and it gives fake good results that won’t work in real life.

✅ Best Way to Do It:

sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Learn & apply scaling on training data
X_test = sc.transform(X_test)  # Apply the same scaling on test data

✔ Train and test data must be scaled the same way for fair results.
✔ Learn from training data, apply to test data.

```
