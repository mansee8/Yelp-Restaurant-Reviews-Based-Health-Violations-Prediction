#Calculating health scores
import pandas as pd

df = pd.read_csv("AllViolations.csv",sep=",")


df['score'] = df['*']+(df['**']*2)+(df['***']*5)

df['normalized_score']= df['score']*100/max(df['score'])

print df['normalized_score'][:5]

df=df.drop('*',axis=1)
df=df.drop('**',axis=1)
df=df.drop('***',axis=1)
df=df.drop('id',axis=1)

print df[:5]

df.to_csv('health_scores.csv')
