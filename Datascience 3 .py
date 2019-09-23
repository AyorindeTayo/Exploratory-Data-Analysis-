
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib


# In[4]:


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head


# In[5]:


get_ipython().run_cell_magic('capture', '', '! pip install seaborn')


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


# list the data types for each column
print(df.dtypes)


# In[8]:


df.corr()


# In[9]:


df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# In[10]:



# Understanding the linear relationship between an individual variable and the price 
# Engine size as pontential predicator variable of price 
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# In[11]:


# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predicator of price since the regression line is almost a perfect diagonal line 
# we can examine the correlation between 'engine-size' and 'price' and see it's approximately 0.87


# In[12]:


df[["engine-size", "price"]].corr()


# In[13]:


sns.regplot(x="highway-mpg", y="price", data=df)


# In[14]:


# As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predicator of price 

# we can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704


# In[15]:


df[['highway-mpg', 'price']].corr()


# In[16]:


# Weak Linear Relationship

# lets see if "Peak-rpm" as a predictor variable of "price"


# In[17]:


sns.regplot(x="peak-rpm", y="price", data=df)


# In[18]:


# peak rpm does not seem like a good predicto of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. Therefore it's not a reliable variable 
# We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616


# In[19]:


df[['peak-rpm','price']].corr()


# In[20]:


# to find correlation between x="stroke", y="price"
# The correlation is 0.0823, the non-diagonal elements of the table
#code:
df[["stroke","price"]].corr()


# In[21]:


# Give the correlation results between "price"stroke" do you expect a linear relationship
#verify your results using the function "regplot()"
# There is a weak corellation between the variable 'stroke' and 'price' as such regression will not work well. We can use "regplot" to demostrate this


# In[22]:


sns.regplot(x="stroke", y="price", data=df)


# In[23]:


# Categorical variables describe a characteristics of a data unit, and are selected from a small group categories. The categorical variables can have the type "object" or "int64". A good way to visualize categorical varibales is by using boxplots 


# In[24]:


sns.boxplot(x="body-style", y="price", data=df)


# In[25]:


# we can se that the distribuiton of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price" 


# In[26]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[27]:


# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine enough to take engine-location as a potential good predictor of price


# In[28]:


# Let's examine "drive-wheels" and "price"
# drive wheels 
sns.boxplot(x="drive-wheels",y="price", data=df)


# In[29]:


# Here we see that the distribution of price between the different drive-wheels could potentially be predictor of price 


# In[30]:


# DESCRIPTIVE STATISTICAL ANALYSIS
df.describe()


# In[31]:


df.describe(include=['object'])


# In[32]:


df['drive-wheels'].value_counts()


# In[33]:


# to convert series to data frame
df['drive-wheels'].value_counts().to_frame()


# In[34]:


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


# In[35]:


# Now we can rename the index to 'drive wheels'

drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[36]:


# we can repaeat the above process for the varibale 'engine-location'
# engine-location as variable 
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace = True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[37]:


# BASICS OF GROUPING 
# lets group the variables of "drive-wheel", we can see 3 categories of drive wheels
df['drive-wheels'].unique()


# In[38]:


df_group_one = df[['drive-wheels','body-style','price']]


# In[39]:


# we can calculate the average price for each of the different categories of data 

# grouping result
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# In[40]:


# grouping results 
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# In[41]:


#Group data is much easier to visualize when it is mae into a pivot table

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[42]:


grouped_pivot = grouped_pivot.fillna(0)   # fill missing values with 0
grouped_pivot


# In[50]:


# grouping results 
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle


# In[51]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


# using heatmap to visualized the relationship between Body style and Price 


# In[53]:


# use grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[54]:


# the heat map plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' in the vertical and horizontal axis respectively. This allows us to visualize how the price is related to 'drive-wheels' and 'body-style'


# In[56]:


# The default labels convey no useful information to us.

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#labels names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#inserts labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[58]:


# corerelation
df.corr()


# In[59]:


from scipy import stats


# In[60]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The pearson Correlation Coefficient is", pearson_coef, "with a P-value of P =", p_value)


# In[61]:


# since the p-value is less than 0.001, the correlation between wheel-base and price is statistically significant, although the lineear relationship is not extremely strong (approximately 0.585)  


# In[62]:


# Horsepower vs Price
# to calculate the pearson correlation coefficient and p-value of 'horsepower' and 'price'


# In[69]:


pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation is", pearson_coef, "with a P-value of P = ", p_value)


# In[64]:


# since the p-value is less than 0.001, the correlation between horsepower and price is stastistically significant, and the linear relationship is quite strong (approximately 0.809, close to 1)


# In[72]:


# to claculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'


# In[73]:


pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P= ", p_value)


# In[ ]:


# Since the p-value is less than 0.001, the correlation between length and price is statistically significant, and the linear realtionship is moderately strong (approximately 0.691)


# In[65]:


# Width and Price
# let's calculate the pearson correlation coefficient and P-value of 'width' and 'price'


# In[71]:


pearson_coef, p_value =stats.pearsonr(df['width'], df['price'])
print("The pearson Correlaion Coeffient is", pearson_coef, "with a P-value of P", p_value)                


# In[ ]:


# since the p-value is less than 0.001, the correlation between width and price is stastically significant, and the linear relationship is quite strong (approx. 0.751)


# In[74]:


# Curb-weight vs Price
# lets calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price':


# In[76]:


pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The pearson Correlation Coefficient is", pearson_coef, "with a P-value of P = ", p_value)


# In[77]:


# since the p-value is less than 0.001, the correlation between curb-weight and price is statistically significant, and linear realtionship is quite strong (approximately 0.834)


# In[78]:


# Highway-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, "with a P-value of P = ", p_value)


# In[ ]:


# Since the p-value is less than 0.001, the correlation between highway-mpg  and price is statistically significant, and the coefficient of approximately -0.705 shows that the relationship is negative and moderately strong


# In[79]:


#ANOVA (Analysis of variance)
# to see if different types 'drive wheels' impact 'price' we group the data


# In[80]:


grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[81]:


df_gptest


# In[96]:


grouped_test2.get_group('4wd')['price']


# In[83]:


# we can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value


# In[98]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)


# In[85]:


# This is a great results, with a large F test score showing a strong correlation and a P value of almost 0 implying almost certain statistical significance. But does this mean all three tested groups are all this highly correlated 


# In[101]:


# separately: fwd and rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)


# In[103]:


# 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P= ", p_val)


# In[105]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])

print("ANOVA results: F=", f_val, ", P=", p_val)


# In[ ]:


# we now have a better idea of how our data looks like and which variables are important to take into account when predicting the car price. 

#Continuous numerical variables: Length, width,Curb-weight, Engine-size, Horsepower, City-mpg, Highway-mpg,wheel-base, Bore, 

# Categorical Variables:
Drive wheels
    

