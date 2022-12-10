#!/usr/bin/env python
# coding: utf-8

# In[53]:


get_ipython().system('pip install neuralprophet')


# In[54]:


import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle


# In[55]:


df.columns


# In[56]:


df = pd.read_csv(r'C:\Users\logi\Downloads\Chennai DEC(1-20-23-22).csv')
df.head()


# In[57]:


df.location


# In[58]:


df.columns


# In[59]:


df.dtypes


# In[60]:


chn = df[df['location']=='Chennai']
chn['date'] = pd.to_datetime(chn['date'])
chn.head()


# In[61]:


chn.dtypes


# In[62]:


plt.plot(chn['date'], chn['temp'])
plt.show()


# In[63]:


chn['Year'] = chn['date'].apply(lambda x: x.year)
chn = chn[chn['Year']<=2022]
plt.plot(chn['date'], chn['temp'])
plt.show()


# In[64]:


data = chn[['date', 'temp']] 
data.dropna(inplace=True)
data.columns = ['ds', 'y'] 
data.head()


# In[65]:


m = NeuralProphet()
m.fit(data, freq='D', epochs=1000)


# In[66]:


future = m.make_future_dataframe(data, periods=1200)
forecast.head()


# In[67]:


plot1 = m.plot(forecast)


# In[68]:


plt2 = m.plot_components(forecast)


# In[69]:


with open('saved_model.pkl', "wb") as f:
    pickle.dump(m, f)


# In[70]:


del m


# In[71]:


with open('saved_model.pkl', "rb") as f:
    m = pickle.load(f)


# In[72]:


future = m.make_future_dataframe(data, periods=1200)
forecast.head()


# In[73]:


plot1 = m.plot(forecast)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




