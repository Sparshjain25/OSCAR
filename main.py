import pandas as pd
import seaborn as sns
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


df1=pd.read_csv(r"oscars-demographics.csv")

f=df1.isnull().sum()/df1.shape[0]*100
drop_column=f[f>20].keys()                                               #To get the names of column having empty % more than 20
df2=df1.drop(columns=drop_column)


                                            #Data is cleaned from null values.

df=df2.loc[:,['birthplace', 'date_of_birth','race_ethnicity', 'year_of_award', 'award']]
print(df.shape)
print(df.head(3))

sns.heatmap(df.isnull())
plt.show()                                                             #Check for null values
print(df.dtypes)


####################        CHANGING DATE(Creating 'idob')            #######################

df['idob']=pd.to_datetime(df['date_of_birth'])
future=pd.DatetimeIndex(df['idob']).year > df['year_of_award']
df.loc[future,'idob']-=timedelta(days=365*100)                         #Changing its datatype
df['idob']=df['idob'].dt.strftime('%d-%m-%Y')


####################            ADDING AGE COLUMN                     #######################

df['award_age']=df['year_of_award']-pd.DatetimeIndex(df['idob']).year


#############                CREATING COUNTRY COLUMN                  #######################

df['country'] = df['birthplace'].str.rsplit(',').str[-1]
df.loc[df['country']=='New York City','country']='USA'
df.loc[df['country']=='Na','country']='USA'
df.loc[df['country'].str.len() == 3,'country']='USA'

data=df.drop(columns=['birthplace','date_of_birth','year_of_award','idob'])


########################################################___DATA___CLEANING___DONE___##########################################


#COUNT OF RACE ORIGIN
df7=df['race_ethnicity'].value_counts()
fig, ax0 = plt.subplots()
df['race_ethnicity'].value_counts().plot(ax=ax0, kind='bar')
ax0.set_title('COUNT OF RACE_ORIGIN')
ax0.set_xlabel('RACE ETHNICITY')
ax0.set_ylabel('Frequency')
plt.show()


#COUNT OF COUNTRY
fig, ax1 = plt.subplots()
df['country'].value_counts().plot(ax=ax1, kind='bar')
ax1.set_title('COUNT CHECK FOR COUNTRY')
ax1.set_xlabel('COUNTRY')
ax1.set_ylabel('Frequency')
plt.show()


#CHECKING AGE W.R.T AWARDS
df7=pd.DataFrame(df.groupby('award')['award_age'].mean().round(decimals=2))
print(df7['award_age'])
fig,ax2 =plt.subplots()
df7.plot(ax=ax2,kind='bar')
ax2.set_title('COMPARISON CHECK')
ax2.set_xlabel('AWARDS')
ax2.set_ylabel('Mean')
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), (p.get_x() , p.get_height()))
plt.show()



#############################################  DATA EXPLORATION DONE   ################################################


#############################################         ENCODING         ################################################

number=LabelEncoder()
data['race_ethnicity']=number.fit_transform(data['race_ethnicity'].astype('str'))
data['country']=number.fit_transform(data['country'].astype('str'))
data['award']=number.fit_transform(data['award'].astype('str'))
Y=data['award']
X=data.drop(columns=['award'])


#############################################       FEATURE SCALING     ###############################################

ms = MinMaxScaler()
X = ms.fit_transform(X)

print(Y)
print(X)

#############################################       BUILDING A MODEL    ###############################################


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.09,random_state=2000,stratify=Y)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

classifier=RandomForestClassifier(n_estimators=100,criterion='gini')                          #RANDOM FOREST CLASSIFIER
classifier.fit(X_train,Y_train)
print("\nAccuracy of RANDOM FOREST:",classifier.score(X_test,Y_test))                         #NAIVE BAYES=37%


########################################   APPLYING K-MEANS   ########################################################

kmeans=KMeans(n_clusters=5)
KMmodel=kmeans.fit(X_train)
labels=KMmodel.labels_
print(labels)


fig = plt.figure(figsize=(10,6))

ax3 = fig.add_subplot(1,2,1, projection='3d')
ax3.set_title('Our Model')
ax3.scatter(X_train[:,0],X_train[:,1],X_train[:,2],c=KMmodel.labels_)

ax3 = fig.add_subplot(1,2,2, projection='3d')
ax3.set_title('Original Data')
ax3.scatter(X_train[:,0],X_train[:,1],X_train[:,2],c=Y_train)

plt.show()

