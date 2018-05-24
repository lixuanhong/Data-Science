import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

ipos = pd.read_excel('SCOOP-Rating-Performance.xls')
print(ipos.head())


# Clean data/Correct data type
ipos.info()
ipos.replace('N/C', 0, inplace=True)
ipos.loc[1061, 'Trade Date'] = '2012-11-20'
ipos['Trade Date'] = pd.to_datetime(ipos['Trade Date'])
ipos['Star Ratings'] = ipos['Star Ratings'].astype('int')
ipos['1st Day % Px Chng '] = ipos['1st Day % Px Chng '] * 100
ipos.info()

# Explore the data
ipos.groupby(ipos['Trade Date'].dt.year)['1st Day % Px Chng '].mean().plot(kind='bar', figsize=(15,10), color='k', title='1st Day Mean IPO Percentage Change')
plt.show()

ipos.groupby(ipos['Trade Date'].dt.year)['1st Day % Px Chng '].median().plot(kind='bar', figsize=(15,10), color='k', title='1st Day Median IPO Percentage Change')
plt.show()

print(ipos['1st Day % Px Chng '].describe())

ipos['1st Day % Px Chng '].hist(figsize=(15, 7), bins=100, color='grey')
plt.show()

ipos['$ Chg Open to Close'] = ipos['$ Change Close'] - ipos['$ Change Opening']
ipos['% Chg Open to Close'] = ipos['$ Chg Open to Close']/ipos['Opening Price'] * 100

# Check abnormal data point and correct
print(ipos.head(10))
print(ipos.sort_values(by=['% Chg Open to Close']).head(10))

ipos.loc[1257, '$ Change Opening'] = .09
ipos.loc[2021, '$ Change Opening'] = .01
ipos.loc[2021, 'Opening Price'] = 11.26

ipos['$ Chg Open to Close'] = ipos['$ Change Close'] - ipos['$ Change Opening']
ipos['% Chg Open to Close'] = ipos['$ Chg Open to Close']/ipos['Opening Price'] * 100

print(ipos['% Chg Open to Close'].describe())

ipos['% Chg Open to Close'].hist(figsize=(15,7), bins=100, color='grey')
plt.show()

# Check stocks trade after 2015-01-01
print(ipos[ipos['Trade Date']>='2015-01-01']['% Chg Open to Close'].describe())

print(ipos[ipos['Trade Date']>='2015-01-01']['% Chg Open to Close'].sum())

# Check winning stocks trade after 2015-01-01
print(ipos[(ipos['Trade Date']>='2015-01-01')&(ipos['$ Chg Open to Close']>0)]
['% Chg Open to Close'].describe())

# Check losing stocks trade after 2015-01-01
print(ipos[(ipos['Trade Date']>='2015-01-01')&(ipos['$ Chg Open to Close']<0)]
['% Chg Open to Close'].describe())

sp = pd.read_csv("SP500.csv")
sp.sort_values('Date', inplace=True)
sp.reset_index(drop=True, inplace=True)
sp.drop(sp.head(1).index, inplace=True) # drop the first row

#Add the change of SP500 index in the past week to the DataFrame
def get_week_chg(ipo_dt):
    try:
        day_ago_idx = sp[sp['Date'] == str(ipo_dt.date())].index[0] - 1
        week_ago_idx = sp[sp['Date'] == str(ipo_dt.date())].index[0] - 8
        chg = (sp.iloc[day_ago_idx]['Close'] - sp.iloc[week_ago_idx]['Close'])/(sp.iloc[week_ago_idx]['Close'])
        return chg * 100
    except:
        print('error', ipo_dt.date())

ipos['SP Week Change'] = ipos['Trade Date'].map(get_week_chg)

#Correct erros
print(ipos[ipos['Trade Date'] == '2015-02-21'])
ipos.loc[310, 'Trade Date'] = pd.to_datetime('2015-05-21')
ipos.loc[311, 'Trade Date'] = pd.to_datetime('2015-05-21')
print(ipos[ipos['Trade Date'] == '2013-11-16'])
ipos.loc[890, 'Trade Date'] = pd.to_datetime('2013-11-06')
print(ipos[ipos['Trade Date'] == '2009-08-01'])
ipos.loc[1389, 'Trade Date'] = pd.to_datetime('2009-08-11')

ipos['SP Week Change'] = ipos['Trade Date'].map(get_week_chg)

def get_cto_chg(ipo_dt):
    try:
        today_open_idx = sp[sp['Date'] == str(ipo_dt.date())].index[0]
        yday_open_idx = sp[sp['Date'] == str(ipo_dt.date())].index[0] - 1
        chg = (sp.iloc[today_open_idx]['Close'] - sp.iloc[yday_open_idx]['Close'])/(sp.iloc[yday_open_idx]['Close'])
        return chg * 100
    except:
        print('error', ipo_dt.date())

ipos['SP Close to Open Chg Pct'] = ipos['Trade Date'].map(get_cto_chg)

# Correct the column name/Rename the column
ipos.columns.values[3] = 'Lead/Joint-Lead Manager'# the 3rd column
print(ipos.head(1))

# Identify and print the main manager
ipos['Lead Mgr'] = ipos['Lead/Joint-Lead Manager'].map(lambda x: x.split('/')[0])
ipos['Lead Mgr'] = ipos['Lead Mgr'].map(lambda x: x.strip())#strip whitespace from left and right sides

for n in pd.DataFrame(ipos['Lead Mgr'].unique(), columns=['Name']).sort_values('Name')['Name']:
    print(n)

# Clean the name of the main Manager
ipos.loc[ipos['Lead Mgr'].str.contains('Edwards'), 'Lead Mgr'] = 'AG Edwards'
ipos.loc[ipos['Lead Mgr'].str.contains('Edwrads'), 'Lead Mgr'] = 'AG Edwards'
ipos.loc[ipos['Lead Mgr'].str.contains('Aegis'), 'Lead Mgr'] = 'Aegis Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('Baird'), 'Lead Mgr'] = 'Baird'
ipos.loc[ipos['Lead Mgr'].str.contains('Banc of America'), 'Lead Mgr'] = 'BofA Merril Lynch'
ipos.loc[ipos['Lead Mgr'].str.contains('Barclays'), 'Lead Mgr'] = 'Barclays'
ipos.loc[ipos['Lead Mgr'].str.contains('Bear'), 'Lead Mgr'] = 'Bear Stearns'
ipos.loc[ipos['Lead Mgr'].str.contains('BoA'), 'Lead Mgr'] = 'BofA Merril Lynch'
ipos.loc[ipos['Lead Mgr'].str.contains('BofA'), 'Lead Mgr'] = 'BofA Merril Lynch'
ipos.loc[ipos['Lead Mgr'].str.contains('Broadband'), 'Lead Mgr'] = 'Broadband Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('Unterberg'), 'Lead Mgr'] = 'C.E. Unterberg Towbin'
ipos.loc[ipos['Lead Mgr'].str.contains('CIBC'), 'Lead Mgr'] = 'CIBC'
ipos.loc[ipos['Lead Mgr'].str.contains('CRT'), 'Lead Mgr'] = 'CRT'
ipos.loc[ipos['Lead Mgr'].str.contains('CS'), 'Lead Mgr'] = 'CSFB'
ipos.loc[ipos['Lead Mgr'].str.contains('Cantor'), 'Lead Mgr'] = 'Cantor Fitzgerald'
ipos.loc[ipos['Lead Mgr'].str.contains('China'), 'Lead Mgr'] = 'China International'
ipos.loc[ipos['Lead Mgr'].str.contains('Cit'), 'Lead Mgr'] = 'Citigroup'
ipos.loc[ipos['Lead Mgr'].str.contains('Cohen'), 'Lead Mgr'] = 'Cohen & Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Cowen'), 'Lead Mgr'] = 'Cowen & Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Craig-Hallum'), 'Lead Mgr'] = 'Craig-Hallum Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('Suisse'), 'Lead Mgr'] = 'CSFB'
ipos.loc[ipos['Lead Mgr'].str.contains('D.A. Davidson'), 'Lead Mgr'] = 'D.A. Davidson & Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Deutsche'), 'Lead Mgr'] = 'Deutsche Bank'
ipos.loc[ipos['Lead Mgr'].str.contains('Donaldson'), 'Lead Mgr'] = 'Donaldson Lufkin & Jenrette'
ipos.loc[ipos['Lead Mgr'].str.contains('EarlyBird'), 'Lead Mgr'] = 'EarlyBird Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('FBR'), 'Lead Mgr'] = 'FBR'
ipos.loc[ipos['Lead Mgr'].str.contains('Feltl'), 'Lead Mgr'] = 'Feltl & Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Ferris'), 'Lead Mgr'] = 'Ferris Baker Watts'
ipos.loc[ipos['Lead Mgr'].str.contains('Friedman'), 'Lead Mgr'] = 'Freidman Billing Ramsey'
ipos.loc[ipos['Lead Mgr'].str.contains('Freidman'), 'Lead Mgr'] = 'Freidman Billing Ramsey'
ipos.loc[ipos['Lead Mgr'].str.contains('GS'), 'Lead Mgr'] = 'Gilford Securities'
ipos.loc[ipos['Lead Mgr'].str.contains('Goldman'), 'Lead Mgr'] = 'Goldman Sachs'
ipos.loc[ipos['Lead Mgr'].str.contains('Gunn'), 'Lead Mgr'] = 'Gunn Allen'
ipos.loc[ipos['Lead Mgr'].str.contains('HCF'), 'Lead Mgr'] = 'HCFP Brenner'
ipos.loc[ipos['Lead Mgr'].str.contains('^I-'), 'Lead Mgr'] = 'I-Bankers'
ipos.loc[ipos['Lead Mgr'].str.contains('J\.P'), 'Lead Mgr'] = 'JP Morgan'
ipos.loc[ipos['Lead Mgr'].str.contains('JMP'), 'Lead Mgr'] = 'JP Morgan'
ipos.loc[ipos['Lead Mgr'].str.contains('JPMorgan'), 'Lead Mgr'] = 'JP Morgan'
ipos.loc[ipos['Lead Mgr'].str.contains('Jeffer'), 'Lead Mgr'] = 'Jefferies'
ipos.loc[ipos['Lead Mgr'].str.contains('Johnson'), 'Lead Mgr'] = 'Johnson Rice'
ipos.loc[ipos['Lead Mgr'].str.contains('Keefe'), 'Lead Mgr'] = 'Keefe Bruyette & Woods'
ipos.loc[ipos['Lead Mgr'].str.contains('Ladenburg'), 'Lead Mgr'] = 'Ladenburg Thalmann'
ipos.loc[ipos['Lead Mgr'].str.contains('MDB'), 'Lead Mgr'] = 'MDB Capital Group LLC'
ipos.loc[ipos['Lead Mgr'].str.contains('Maxi'), 'Lead Mgr'] = 'Maxim Group'
ipos.loc[ipos['Lead Mgr'].str.contains('Merril'), 'Lead Mgr'] = 'Merrill Lynch'
ipos.loc[ipos['Lead Mgr'].str.contains('Morgan Stan'), 'Lead Mgr'] = 'Morgan Stanley'
ipos.loc[ipos['Lead Mgr'].str.contains('Needham'), 'Lead Mgr'] = 'Needham Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Oppenhe'), 'Lead Mgr'] = 'Oppenheimer'
ipos.loc[ipos['Lead Mgr'].str.contains('Pali'), 'Lead Mgr'] = 'Pali Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('Paulson'), 'Lead Mgr'] = 'Paulson Investment Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Piper'), 'Lead Mgr'] = 'Piper Jaffray'
ipos.loc[ipos['Lead Mgr'].str.contains('Roberston'), 'Lead Mgr'] = 'Robertson Stephens'
ipos.loc[ipos['Lead Mgr'].str.contains('Rodman'), 'Lead Mgr'] = 'Rodman & Renshaw'
ipos.loc[ipos['Lead Mgr'].str.contains('Roth'), 'Lead Mgr'] = 'Roth Capital'
ipos.loc[ipos['Lead Mgr'].str.contains('SANDLER'), 'Lead Mgr'] = 'Sandler O\'Neil + Partners'
ipos.loc[ipos['Lead Mgr'].str.contains('Sandler'), 'Lead Mgr'] = 'Sandler O\'Neil + Partners'
ipos.loc[ipos['Lead Mgr'].str.contains('SG Cowen'), 'Lead Mgr'] = 'SG Cowen'
ipos.loc[ipos['Lead Mgr'].str.contains('Stifel'), 'Lead Mgr'] = 'Stifel Nicolaus Weisel'
ipos.loc[ipos['Lead Mgr'].str.contains('SunTrust'), 'Lead Mgr'] = 'SunTrust Robinson Humphrey'
ipos.loc[ipos['Lead Mgr'].str.contains('Thomas'), 'Lead Mgr'] = 'Thomas Weisel'
ipos.loc[ipos['Lead Mgr'].str.contains('UBS'), 'Lead Mgr'] = 'UBS'
ipos.loc[ipos['Lead Mgr'].str.contains('W\.R\.'), 'Lead Mgr'] = 'W.R. Hambrecht + Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('WR'), 'Lead Mgr'] = 'W.R. Hambrecht + Co.'
ipos.loc[ipos['Lead Mgr'].str.contains('Wachovia'), 'Lead Mgr'] = 'Wachovia'
ipos.loc[ipos['Lead Mgr'].str.contains('Wedbush'), 'Lead Mgr'] = 'Wedbush Morgan'
ipos.loc[ipos['Lead Mgr'].str.contains('William Blair'), 'Lead Mgr'] = 'William Blair'
ipos.loc[ipos['Lead Mgr'].str.contains('Wunderlich'), 'Lead Mgr'] = 'Wunderlich'

for n in pd.DataFrame(ipos['Lead Mgr'].unique(), columns=['Name']).sort_values('Name')['Name']:
    print(n)

# Add a few more characteristics
ipos['Total Underwriters'] = ipos['Lead/Joint-Lead Manager'].map(lambda x: len(x.split('/')))

ipos['Week Day'] = ipos['Trade Date'].dt.dayofweek.map({0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
ipos['Month'] = ipos['Trade Date'].map(lambda x: x.month)
ipos['Month'] = ipos['Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
11: 'Nov', 12: 'Dec'})

ipos['Gap Open Pct'] = (ipos['$ Change Opening']/ipos['Opening Price']) * 100
ipos['Open to Close Pct'] = (ipos['$ Change Close'] - ipos['$ Change Opening'])/ipos['Opening Price'] * 100

# Convert preliminary data to matrix for building statistical model
X = dmatrix('Month + Q("Week Day") + Q("Total Underwriters") + Q("Gap Open Pct") + Q("$ Change Opening") +\
Q("SP Close to Open Chg Pct") + Q("SP Week Change")', data=ipos, return_type='dataframe')

# Use 2015's data for testing and the previous years's data for training
X_train, X_test = X[406:], X[206:406]
y_train = ipos['$ Chg Open to Close'][406:].map(lambda x: 1 if x >=1 else 0)
y_test = ipos['$ Chg Open to Close'][206:406].map(lambda x: 1 if x>=1 else 0)

clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(ipos[(ipos['Trade Date']>='2015-01-16') & (ipos['Trade Date']<='2015-12-18')]['$ Chg Open to Close'].describe())

#Predict result
pred_label = clf.predict(X_test)
results=[]
for p1, t1, idx, chg in zip(pred_label, y_test, y_test.index, ipos.ix[y_test.index]['$ Chg Open to Close']):
    if p1 == t1:
        results.append([idx, chg, p1, t1, 1])
    else:
        results.append([idx, chg, p1, t1, 0])

rf = pd.DataFrame(results, columns=['index', '$ chg', 'predicted', 'actual', 'correct'])
print(rf)

print(rf[rf['predicted']==1]['$ chg'].describe())

fig, ax = plt.subplots(figsize=(15, 10))
rf[rf['predicted']==1]['$ chg'].plot(kind='bar')
ax.set_title('Model Predicted Buys', y=1.01)
ax.set_ylabel('$ Change Open to Close')
ax.set_xlabel('Index')
plt.show()

#Test the Model/ The accuracy of the model decreses
X_train, X_test = X[406:], X[206:406]
y_train = ipos['$ Chg Open to Close'][406:].map(lambda x: 1 if x >=.25 else 0)
y_test = ipos['$ Chg Open to Close'][206:406].map(lambda x: 1 if x>=.25 else 0)
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

pred_label = clf.predict(X_test)
results=[]
for p1, t1, idx, chg in zip(pred_label, y_test, y_test.index, ipos.ix[y_test.index]['$ Chg Open to Close']):
    if p1 == t1:
        results.append([idx, chg, p1, t1, 1])
    else:
        results.append([idx, chg, p1, t1, 0])

rf = pd.DataFrame(results, columns=['index', '$ chg', 'predicted', 'actual', 'correct'])
print(rf[rf['predicted']==1]['$ chg'].describe())

# Add the 2014 data to the test datasets
X_train, X_test = X[694:], X[206:694]
y_train = ipos['$ Chg Open to Close'][694:].map(lambda x: 1 if x >=1 else 0)
y_test = ipos['$ Chg Open to Close'][206:694].map(lambda x: 1 if x>=1 else 0)
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

pred_label = clf.predict(X_test)
results=[]
for p1, t1, idx, chg in zip(pred_label, y_test, y_test.index, ipos.ix[y_test.index]['$ Chg Open to Close']):
    if p1 == t1:
        results.append([idx, chg, p1, t1, 1])
    else:
        results.append([idx, chg, p1, t1, 0])

rf = pd.DataFrame(results, columns=['index', '$ chg', 'predicted', 'actual', 'correct'])
print(rf[rf['predicted']==1]['$ chg'].describe())

fv = pd.DataFrame(X_train.columns, clf.coef_.T).reset_index()
fv.columns = ['Coef', 'Feature']
fv.sort_values('Coef', ascending=0).reset_index(drop=True)
print(fv)
print(fv[fv['Feature'].str.contains('Week Day')])
print(ipos[ipos['Lead Mgr'].str.contains('Keegan|Towbin')])

clf_rf = RandomForestClassifier(n_estimators=1000)
clf_rf.fit(X_train, y_train)
f_importances = clf_rf.feature_importances_
f_names = X_train
f_std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
zz = zip(f_importances, f_names, f_std)
zzs = sorted(zz, key=lambda x: x[0], reverse=True)
imps = [x[0] for x in zzs[:20]]
labels = [x[1] for x in zzs[:20]]
errs = [x[2] for x in zzs[:20]]
plt.subplots(figsize=(15,10))
plt.bar(range(20), imps, color="r", yerr=errs, align="center")
plt.xticks(range(20), labels, rotation=-70)
plt.show()
