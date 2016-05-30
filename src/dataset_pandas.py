import json
from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
#from stemming.porter2 import stem
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


#get all restaurants and the date they were reviewed for health inspections
all_violations = pd.read_csv("AllViolations.csv",sep=",")
all_restaurants = all_violations[:]["restaurant_id"]
all_review_dates = all_violations[:]["date"]
all_severity_score_1 = all_violations[:]["*"]
all_severity_score_2 = all_violations[:]["**"]
all_severity_score_3 = all_violations[:]["***"]
all_health_ins_date=[None]*len(all_review_dates)
for i in range(len(all_review_dates)):
    all_health_ins_date[i]=datetime.strptime(all_review_dates[i], '%Y-%m-%d')

#looking into main review dataset
data = []
count=0
with open("yelp_boston_academic_dataset/yelp_academic_dataset_review.json") as f:
    for line in f:
        count=count+1
        if count==235214:
            break
        data.append(json.loads(line))

#create a dataframe with 4 columns
columns= ['restaurant_id','health_inspection_date','all_reviews_text','*','**','***']
final_table = pd.DataFrame(columns=columns)
final_table = final_table.fillna(0)

#add values to final_table
final_table['restaurant_id']=all_restaurants
final_table['health_inspection_date']=all_health_ins_date
final_table['*']=all_severity_score_1
final_table['**']=all_severity_score_2
final_table['***']=all_severity_score_3

all_business_id=[]
all_reviews_global=[]
all_restaurant_id=[]

rest_to_bus = pd.read_csv("restaurant_ids_to_yelp_ids.csv")
#get all business IDs
for i in range(len(data)):
    business_id = data[i]["business_id"]
    all_reviews_restaurant=[]
    #now append the business id and the review
    all_business_id.append(business_id)


corresponding_rest_id=[]
unique_buss_id =set()
#map business id to restaurant id
for bus_id in range(len(all_business_id)):
    #print all_business_id[bus_id]
    a= rest_to_bus[rest_to_bus["yelp_id_0"]==all_business_id[bus_id]]
    b= rest_to_bus[rest_to_bus["yelp_id_1"]==all_business_id[bus_id]]
    c= rest_to_bus[rest_to_bus["yelp_id_2"]==all_business_id[bus_id]]
    rest_id=''
    if len(a)>0 and len(a.get_values()[0][0])>0:
        rest_id=a.get_values()[0][0]
    elif len(b)>0 and  len(b.get_values()[0][0])>0:
        rest_id=b.get_values()[0][0]
    elif len(c)>0 and len(c.get_values()[0][0])>0:
        rest_id=c.get_values()[0][0]
    if rest_id=='':
        print all_business_id[bus_id]
        print bus_id
        unique_buss_id.add(all_business_id[bus_id])
    corresponding_rest_id.append(rest_id)

#unique_buss_id has the business ID whose rest ID are not present in mapping

print "rest id corr to yelp id"
print corresponding_rest_id[:5]
print all_business_id[:5]
#at this point we have every row of review's rest id and yelp ID
#now we will combine reviews for the restaurant if it occurs before the health ins date


for i in range(len(corresponding_rest_id)):
    if corresponding_rest_id[i]=='':
        continue
    business_id = data[i]["business_id"]
    all_reviews_restaurant=[]
    if all_business_id.__contains__(business_id):
        #check that review date is before the health inspection date
        date_review= datetime.strptime(data[i]["date"], '%Y-%m-%d')
        #get the health check date
        row=np.where(final_table.loc[final_table['restaurant_id'] == corresponding_rest_id[i]])[0]
        health_ins_dat = datetime.today()
        for r in row:

            health_ins_dat=final_table['health_inspection_date'][r]
            if(date_review<=health_ins_dat):
                row=r
                break

        if(date_review<=health_ins_dat):
            all_reviews= final_table['all_reviews_text'][row]
            final_table['all_reviews_text'][row] = []
            #print all_reviews
            if all_reviews.__str__()=='nan':
                final_table['all_reviews_text'][row]=[data[i]["text"]]
            else:
                print "row"
                print i
                all_reviews.append(data[i]["text"])
                final_table['all_reviews_text'][row]=all_reviews


#write to a csv file
final_table.to_csv("with_reviews.csv")
#join all reviews

#if laptop crashes : lets start with csv loading and continue
final_table = pd.read_csv("with_reviews.csv",sep=",")

final_table["all_reviews_string"]=''
snowball_stemmer = SnowballStemmer("english")
wordnet_lemmatizer = WordNetLemmatizer()

for i in range(len(final_table)):
    review = final_table["all_reviews_text"][i]
    #print review
    if review.__str__()!='nan':
        str1 = ''.join(review)
        #lemmetize it
        #wordnet_lemmatizer.lemmatize
        documents = " ".join([wordnet_lemmatizer.lemmatize(word) for word in str1.split(" ")])
        #stem it
        documents = " ".join([snowball_stemmer.stem(word) for word in documents.split(" ")])
        print documents
        final_table["all_reviews_string"][i] = documents


stop = stopwords.words('english')
stop.remove(u'no')
stop.remove(u'nor')
stop.remove(u'not')
stop.append(u'let')
stop.append(u'anyway')
stop.append(u'else')
stop.append(u'maybe')
stop.append(u'however')
stop.append(u'00')
stop.append(u'10')
stop.append(u'11')
stop.append(u'12')
stop.append(u'15')
stop.append(u'20')
stop.append(u'25')
stop.append(u'30')
stop.append(u'40')
stop.append(u'50')
vectorizer = TfidfVectorizer(stop_words=stop, use_idf=True,max_features=1500)
#ngram ,ngram_range=(1,7)


X = vectorizer.fit_transform(final_table["all_reviews_string"])

#top 50 features
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 50
top_features = [features[i] for i in indices[:top_n]]
#print top_features
final_table['top_50'][i]=top_features

text_processing_output= pd.DataFrame(data=X.todense(), columns=vectorizer.get_feature_names()).iloc[:,:]


op2= text_processing_output

op2['restaurant_id'] = pd.Series(final_table["restaurant_id"])
op2['health_inspection_date'] = pd.Series(final_table["health_inspection_date"])
op2['*'] = pd.Series(final_table["*"])
op2['**'] = pd.Series(final_table["**"])
op2['***'] = pd.Series(final_table["***"])
op2.to_csv("Data_1500.csv")
