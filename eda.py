# %%
import pandas as pd


# %%
import seaborn as sns

# %%
import numpy as np

# %%
import matplotlib.pyplot as plt

# %%
kashti=sns.load_dataset("titanic")

# %%
kashti.to_csv("kashti.csv")

# %%
kashti.info()

# %%
kashti.head()

# %%
kashti.shape

# %%
kashti.describe()

# %% [markdown]
# # Unique values

# %%
kashti.nunique()

# %% [markdown]
# # Column names

# %%
kashti.columns

# %%
kashti["sex"].unique()

# %%
kashti["who"].unique()

# %% [markdown]
# # Clean the data

# %%
kashti.isnull()

# %%
kashti.isnull().sum()

# %%
kashti.drop(["deck"],axis=1)

# %%
ks_clean=kashti.drop(["deck"],axis=1)
ks_clean

# %%
ks_clean.isnull().sum()

# %%
ks=ks_clean.dropna()

# %%
ks

# %%
ks["age"].value_counts()

# %%
ks["who"].value_counts()

# %%
kashti.describe()

# %%
ks.describe()

# %%
ks.columns

# %%
sns.boxplot(x="sex",y="age",data=ks)

# %%
sns.boxplot(y="age",data=ks)

# %%
sns.distplot(ks["age"])

# %%
sns.distplot(ks["age"])
plt.show()

# %% [markdown]
# # Outliers removal

# %%
ks["age"].mean()

# %%
ks=ks[ks["age"]<68]
ks

# %%
ks["age"].mean()

# %%
ks=ks[ks["age"]<60]

# %%
ks["age"].mean()

# %%
sns.distplot(ks["age"])

# %%
sns.boxplot(y="age",data=ks)

# %%
ks.boxplot()

# %%
sns.boxplot(y="fare",data=ks)

# %%
ks=ks[ks["fare"]<300]
ks

# %%
sns.distplot(ks["fare"])

# %%
ks.hist()

# %%
pd.value_counts(ks["survived"]).plot.bar()

# %%
pd.value_counts(ks["sex"]).plot.bar()

# %%
pd.value_counts(ks["class"]).plot.bar()

# %%
ks.groupby(["sex","class"]).mean()

# %% [markdown]
# #Correlation

# %%
Corr=ks.corr()
Corr

# %%
sns.heatmap(Corr)

# %%
sns.heatmap(Corr,annot=True)

# %%
sns.relplot(x="age",y="fare", data=ks)

# %%
sns.relplot(x="age",y="fare", hue="sex", data=ks)

# %%
sns.relplot(x="age",y="fare", hue="class", data=ks)

# %%
sns.catplot(x="sex",y="fare", hue="sex", data=ks, kind="box")

# %%
sns.catplot(x="sex",y="age", hue="sex", data=ks, kind="box")

# %% [markdown]
# # Log transformation

# %%
ks_log["fare_log"]=np.log(ks["fare"])

# %%
ks.head()


