#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#      <h1>OASIS INFOBYTE </h1>
#     <h1>Author:-Abhilash Purohit</h1>
#     <h1>Task-1 </h1>
#     <h1>IRIS FLOWER CLASSIFICATION </h1>
# </div>

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Load the Iris dataset (replace 'path_to_iris_dataset.csv' with the actual path)
iris_data = pd.read_csv(r"C:\Users\dubey\OneDrive\Desktop\Iris.csv")

# Explore the dataset
print(iris_data.head())


# In[4]:


# Split the data into features (X) and labels (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']


# In[5]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


# Choose a model (Logistic Regression in this case)
model = LogisticRegression()


# In[8]:


# Make predictions
y_pred = model.predict(X_test)


# In[9]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix_df = confusion_matrix(y_test, y_pred)
classification_report_df = classification_report(y_test, y_pred)


# In[10]:


# Print results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_matrix_df}')
print(f'Classification Report:\n{classification_report_df}')


# <div style="text-align: center;">
#     <h1>Task-2 </h1>
#     <h1>UNEMPLOYMENT ANALYSIS WITH PYTHON</h1>
# </div>

# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv(r"C:\Users\dubey\OneDrive\Desktop\Unemployment in India.csv")

# Display the first few rows to understand the structure of the dataset
print(df.head())


# In[46]:


# Display the column names in your dataset
print(df.columns)


# In[47]:


# Remove leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

# Check the column names again
print(df.columns)


# In[48]:


# Check for missing values
print(df.isnull().sum())

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])


# In[49]:


# Plot unemployment rate over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df, hue='Region')
plt.title('Unemployment Rate Over Time')
plt.show()


# In[50]:


# Descriptive statistics
print(df[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].describe())


# In[51]:


# Correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# <div style="text-align: center;">
#     <h1>Task-3 </h1>
#   <h1>CAR PRICE PREDICTION WITH MACHINE LEARNING</h1>
# </div>

# In[52]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[63]:


# Load the dataset
df = pd.read_csv(r"C:\Users\dubey\OneDrive\Desktop\car data.csv")  # Replace with the actual file path or URL

# Display the first few rows to understand the structure of the dataset
df.head()



# In[64]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']):
    df[column] = label_encoder.fit_transform(df[column])

df.head()


# In[65]:


x=df.drop(["Present_Price"],axis=1)
x.head()


# In[66]:


y=df[["Present_Price"]].copy()
y


# In[71]:


x["intercept"]=1


# In[73]:


x=x[["intercept","Car_Name","Year","Selling_Price","Driven_kms","Fuel_Type","Selling_type","Transmission","Owner"]]


# In[74]:


x


# In[75]:


x= x.values
y = y.values


# In[76]:


x


# In[77]:


y


# In[78]:


## split the data into training and testing test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[79]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[80]:


## X_train Tronspose 
x_train_T = x_train.T
x_train_T


# In[81]:


# finding of beta head B = inv(x_train.T*x_train)*x_train*y_train 
B =  np.linalg.pinv(x_train_T @ x_train) @ x_train_T @ y_train
B.shape


# In[82]:


## finding of predications 
y_pred = x_test @ B


# In[83]:


y_pred


# In[84]:


## shape of y_test,y_pred
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)


# In[85]:


y_test = y_test.ravel()  # or y_test.flatten()
y_pred = y_pred.ravel()  # or y_pred.flatten()


# In[86]:


## prediction of dataframe 
pred_price = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[87]:


pred_price


# In[88]:


## mean squared error and correleation
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R2 Score:', r2)


# In[89]:


## Residual sum of square and mean squared error 
rss = 0
for i in range(len(y_test)):
    rss += (y_test[i]-y_pred[i])**2
print("Residual Sum of Squares: ",rss)
mse = rss/len(y_test)
print("Mean Squared Error: ",mse)


# In[90]:


## Data Visualazation 
import seaborn as sns
sns.pairplot(df)


# <div style="text-align: center;">
#     <h1>Task-4 </h1>
#  <h1>EMAIL SPAM DETECTION WITH MACHINE LEARNING</h1>
# </div>

# In[91]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[105]:


# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv(r"C:\Users\dubey\OneDrive\Desktop\spam.csv", encoding='latin1')


# In[106]:


df.head()


# In[107]:


# Example: Remove any HTML tags and convert text to lowercase
df['v2'] = df['v2'].str.replace('<[^<]+?>', '', regex=True)
df['v2'] = df['v2'].str.lower()


# In[108]:


X = df['v1']
y = df['v2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[109]:


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[110]:


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# In[111]:


# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[113]:


# Example: Make predictions for new emails
new_emails = ["Congratulations! You've won a prize.", 'Meeting at 3 PM in the conference room.']
new_emails_vectorized = vectorizer.transform(new_emails)

predictions = model.predict(new_emails_vectorized)
print(f'Predictions for new emails: {predictions}')


# <div style="text-align: center;">
#     <h1>Task-5</h1>
#  <h1>SALES PREDICTION USING PYTHON</h1>
# </div>

# In[128]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[129]:


data=pd.read_csv("C:\\Users\\dubey\\OneDrive\\Desktop\\Advertising.csv",index_col='Unnamed: 0')


# In[117]:


data.head()


# In[118]:


data.isnull().sum()


# In[119]:


data.describe()


# In[120]:


sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=0.7)
plt.show()


# In[121]:


sns.heatmap(data.corr(), annot=True)
plt.show()


# In[122]:


# Define features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# In[123]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[124]:


# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[125]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[126]:


# Model evaluation
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))


# In[127]:


# You can now use the trained model to predict sales for new data
new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [10]})
predicted_sales = model.predict(new_data)
print('Predicted Sales:', predicted_sales)

