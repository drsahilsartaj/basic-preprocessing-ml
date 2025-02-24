# basic-preprocessing-ml

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

