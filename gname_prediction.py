import pandas as pd
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

gtd = pd.read_csv('globalterrorismdb_0616dist.csv')

features = ['gname',
'iyear',
'imonth',
'iday',
'extended',
'country',
'region',
'crit1',
'crit2',
'crit3',
'doubtterr',
'multiple',
'success',
'suicide',
'attacktype1',
'targtype1',
'natlty1',
'guncertain1',
'weaptype1',
'nkill',
'nkillter',
'nwound',
'nwoundte',
'property',
'ishostkid',
'INT_LOG',
'INT_IDEO',
'INT_MISC',
'INT_ANY']

gtd_features = gtd[features]

# cleaning and data munging
gtd_features.natlty1.fillna(gtd_features.country, inplace=True) #if nationality of target is missing assume same as country
gtd_features.guncertain1.fillna(0, inplace=True) #assume missing perputrators not suspected
gtd_features.nkill = gtd_features.groupby("country").nkill.transform(lambda x: x.fillna(math.ceil(x.mean()))) # if missing assume nkill is average nkill for country
gtd_features.nkill.fillna(int(gtd_features.nkill.mode()), inplace=True) # if no data for country replace with mode of all data
gtd_features.nkillter = gtd_features.groupby("country").nkillter.transform(lambda x: x.fillna(math.ceil(x.mean()))) # if missing assume nkillter is average nkillter for country
gtd_features.nkillter.fillna(int(gtd_features.nkillter.mode()), inplace=True) # if no data for country replace with mode of all data
gtd_features.nwound = gtd_features.groupby("country").nwound.transform(lambda x: x.fillna(math.ceil(x.mean()))) # if missing assume nwound is average nwound for country
gtd_features.nwound.fillna(int(gtd_features.nwound.mode()), inplace=True) # if no data for country replace with mode of all data
gtd_features.nwoundte = gtd_features.groupby("country").nwoundte.transform(lambda x: x.fillna(math.ceil(x.mean()))) # if missing assume nwoundter is average nwoundter for country
gtd_features.nwoundte.fillna(int(gtd_features.nwoundte.mode()), inplace=True) # if no data for country replace with mode of all data
gtd_features.ishostkid.fillna(-9, inplace=True) # if data missing assign as unknown

d = defaultdict(LabelEncoder)
gtd_features_label_encode = gtd_features.apply(lambda x: d[x.name].fit_transform(x))

X = gtd_features_label_encode.drop('gname', 1)
y = gtd_features_label_encode.gname

# training data of known group names
y_train = y.loc[gtd_features.gname != 'Unknown']
X_train = X.ix[y_train.index.values]

# testing data of unknown groups to be predicted
X_test = X.ix[y.loc[gtd_features.gname == 'Unknown'].index.values]


# random forest classifier
forest = RandomForestClassifier(n_estimators = 100) #more estimators desirable
forest = forest.fit(X_train, y_train)
#joblib.dump(forest, 'rf_n_10.pkl') 
#joblib.load('rf_n_10.pkl') 

#k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)
scores = cross_val_score(forest, X_train, y_train, cv = kf)

# predict unknown group names
prediction = forest.predict(X_test)
prediction_gname = pd.DataFrame(prediction, columns=['gname'])

prediction = pd.concat([pd.DataFrame(prediction_gname), prediction.reset_index(drop=True)], axis=1)

prediction.apply(lambda x: d[x.name].inverse_transform(x)).to_csv('prediction.csv') #decode labels and output prediction to csv
