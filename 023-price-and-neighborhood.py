#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Predicting Price with Neighborhood</strong></font>

# In[20]:


import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)



# # Prepare Data

# ## Import

# In[22]:


def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Extract "Neighborhood"
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    
    df.drop(columns="place_with_parent_names", inplace=True)

    return df


# In[24]:


files = glob("data/buenos-aires-real-estate-*.csv")
files = ['data/buenos-aires-real-estate-1.csv',
 'data/buenos-aires-real-estate-2.csv',
 'data/buenos-aires-real-estate-3.csv',
 'data/buenos-aires-real-estate-4.csv',
 'data/buenos-aires-real-estate-5.csv']
files


# In[25]:

# In[27]:


frames = []
for file in files:
    df = wrangle(file)
    frames.append(df)


# In[30]:


df = pd.concat(frames, ignore_index=True)
df.head()



# ## Split

# In[35]:


target = "price_aprox_usd"
features = ["neighborhood"]
X_train = df[features]
y_train = df[target]



# In[38]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

print("Mean apt price:", mean_absolute_error(y_train, y_pred_baseline))

print("Baseline MAE:", y_mean)


# In[40]:


ohe = OneHotEncoder(use_cat_names=True)
ohe.fit(X_train)
XT_train = ohe.transform(X_train)
print(XT_train.shape)
XT_train.head()


# In[50]:


model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)
model.fit(X_train, y_train)


# In[44]:


X_train.head()



# ## Evaluate

# In[47]:


y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# In[51]:


X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


# In[54]:


intercept = model.named_steps["ridge"].intercept_.round()
coefficients = model.named_steps["ridge"].coef_.round()
print("coefficients len:", len(coefficients))
print(coefficients[:5])  # First five coefficients


# In[56]:


feature_names = model.named_steps["onehotencoder"].get_feature_names()
print("features len:", len(feature_names))
print(feature_names[:5])  # First five feature names

# In[60]:


feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp.head()


# In[62]:


print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")


# In[67]:


feat_imp.sort_values(key=abs).tail(15).plot(
    kind="barh",
    xlabel="Importance [USD]",
    ylabel="Features",
    title="Feature Importance for Apaetment Price");


