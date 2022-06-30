#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Predicting Price with Size</strong></font>

# In[1]:


import warnings

import matplotlib.pyplot as plt
import pandas as pd
import wqet_grader
from IPython.display import VimeoVideo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted
# ## Import

# In[4]:


def wrangle(filepath):
    df = pd.read_csv(filepath)
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]
    return df


# In[6]:


df = wrangle("data/buenos-aires-real-estate-1.csv")
print("df shape:", df.shape)
df.head()

# ## Explore

# In[11]:


plt.hist(df["surface_covered_in_m2"])
plt.xlabel("Area [sq meters]")
plt.title("Distribution of Apartment Sizes")


# In[13]:


#df["surface_covered_in_m2"].describe()
df.describe()["surface_covered_in_m2"]


# In[17]:


plt.scatter(x=df["surface_covered_in_m2"], y=df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Price vs. Area")


# In[20]:


features = ["surface_covered_in_m2"]
X_train = df[features]
print(X_train.shape)
X_train.head()


# In[23]:


target = "price_aprox_usd"
y_train = df[target]
print(y_train.shape)
y_train.head()

# In[25]:


y_mean = y_train.mean()
y_mean

# In[27]:


y_pred_baseline = [y_mean] * len(y_train)
len(y_pred_baseline)


# In[29]:


plt.plot(X_train, y_pred_baseline, color = "orange", label = "Baseline Model")
plt.scatter(X_train, y_train)
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Buenos Aires: Price vs. Area")
plt.legend();



# In[30]:


mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price:", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))



# ## Iterate
# In[31]:


model = LinearRegression()

# In[33]:


model.fit(X_train, y_train)

# ## Evaluate

# In[35]:


y_pred_training = model.predict(X_train)
y_pred_training[:5]

# In[37]:


mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# In[38]:


X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()



# In[41]:


intercept = model.intercept_
print("Model Intercept:", round(intercept,2))
assert any([isinstance(intercept, int), isinstance(intercept, float)])


# In[43]:


coefficient = model.coef_[0]
print('Model coefficient for "surface_covered_in_m2":', coefficient)
assert any([isinstance(coefficient, int), isinstance(coefficient, float)])


# In[45]:


print(f"apt_price = {intercept} + {coefficient} * surface_covered")


# In[46]:


plt.plot(X_train, model.predict(X_train), color="r", label="Linear Model")
plt.scatter(X_train, y_train)
plt.xlabel("surface covered [sq meters]")
plt.ylabel("price [usd]")
plt.legend();
