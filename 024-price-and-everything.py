#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Predicting Price with Size, Location, and Neighborhood</strong></font>

# In[1]:


import warnings
from glob import glob

import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted


# ## Import

# In[3]:


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

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    # Drop features with high NaN values
    df.drop(columns=["floor","expenses"], inplace=True)
    
    # Drop low or high cardinality columns
    df.drop(columns=["operation","property_type","currency","properati_url"], inplace=True)

    # Drop leaky columns
    df.drop(columns=['price',
                     'price_aprox_local_currency',
                     'price_per_m2',
                     'price_usd_per_m2'], inplace=True)
    # Drop columns with Multicollinearity
    df.drop(columns=["surface_total_in_m2","rooms"], inplace=True)
    
    return df


# In[5]:


files = glob("data/buenos-aires-real-estate-*.csv")
files = ['data/buenos-aires-real-estate-1.csv',
 'data/buenos-aires-real-estate-2.csv',
 'data/buenos-aires-real-estate-3.csv',
 'data/buenos-aires-real-estate-4.csv',
 'data/buenos-aires-real-estate-5.csv']
files


# In[8]:


frames = [wrangle(file) for file in files]
frames[0].head()

# In[11]:


df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head()


# In[14]:


df.isnull().sum() / len(df) *100


# In[17]:


df.select_dtypes("object").head()


# In[18]:


df.select_dtypes("object").nunique()


# In[21]:


sorted(df.columns)


# In[24]:


corr = df.select_dtypes("number").drop(columns="price_aprox_usd").corr()
sns.heatmap(corr)


# In[25]:


df.info()

# ## Split Data

# In[27]:


target = "price_aprox_usd"
#features = ["surface_covered_in_m2","lat","lon","neighborhood"]
X_train = df[["surface_covered_in_m2","lat","lon","neighborhood"]]
y_train = df[target]

# In[30]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

print("Mean apt price:", mean_absolute_error(y_train, y_pred_baseline))

print("Baseline MAE:", y_mean)


# ## Iterate

# In[31]:


model = make_pipeline(
    OneHotEncoder(),
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)

# In[34]:


y_pred_training = model.predict(X_train)
print("Training MAE:", mean_absolute_error(y_train, y_pred_training))

# In[35]:


X_test = pd.read_csv("data/buenos-aires-test-features.csv")
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


# In[37]:


def make_prediction(area, lat, lon, neighborhood):
    data = {
        "surface_covered_in_m2": area,
        "lat":lat,
        "lon":lon,
        "neighborhood":neighborhood
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted apartment price: ${prediction}"


# In[38]:


make_prediction(110, -34.60, -58.46, "Villa Crespo")


# In[39]:


interact(
    make_prediction,
    area=IntSlider(
        min=X_train["surface_covered_in_m2"].min(),
        max=X_train["surface_covered_in_m2"].max(),
        value=X_train["surface_covered_in_m2"].mean(),
    ),
    lat=FloatSlider(
        min=X_train["lat"].min(),
        max=X_train["lat"].max(),
        step=0.01,
        value=X_train["lat"].mean(),
    ),
    lon=FloatSlider(
        min=X_train["lon"].min(),
        max=X_train["lon"].max(),
        step=0.01,
        value=X_train["lon"].mean(),
    ),
    neighborhood=Dropdown(options=sorted(X_train["neighborhood"].unique())),
);
