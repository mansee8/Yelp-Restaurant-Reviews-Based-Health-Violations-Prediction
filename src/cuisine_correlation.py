#extract all cuisines
#create a matrix with restaurat id as rows and cuisines as cols
#check reviews to see the presence of cuisine in reviews and the final health score
import pandas as pd
import numpy as np


pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

df = pd.read_json('yelp_boston_academic_dataset/yelp_academic_dataset_business.json',orient='columns')
df_categories_rest = df[['categories','business_id']]

#get unique categories
df2=df['categories']
df2 = df2.values.flatten().tolist() #form a list
df2 = [item for sublist in df2 for item in sublist] #flatten the list
df2 = list(set(df2)) #remove duplicates


#form a matrix with unique cat and business id
df = pd.DataFrame(np.random.rand(len(df_categories_rest),len(df2)),index=df_categories_rest['business_id'],columns=df2)
print df.shape
#dataframe initialized with 0 values
df[:] = 0

#one hot encoding
for i in range(len(df_categories_rest)):
    val = df_categories_rest['categories'][i]
    for v in val:
        df[v][i]=1

df.to_csv('cuisine_one_hot_encoding.csv')