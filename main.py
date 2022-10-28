import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from math import sqrt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


data = pd.read_csv("OnlineNewsPopularity.csv")
data = data.rename(columns=lambda x: x.strip())
data
data[data.isnull()]
data.info()
data.describe()
data.corr()
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')
corr.shares.describe()
corr[corr.shares > 0.02].index
ax1 = data.plot.scatter(y='num_hrefs', x='shares', c='DarkBlue')
for i in list(corr[corr.shares > 0.02].index):
    print(i)
plt.figure(figsize=(12, 6))
ax = sns.scatterplot(data=data, x="num_hrefs", y="shares")
plt.title("")
plt.show()
for i in list(corr[corr.shares > 0.02].index):
    print(i)
    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(data=data, x=i, y="shares")
    plt.title(i)
    plt.show()

temp_data = data[data['shares'] <= 100000]

dff = data[["shares", "rate_positive_words", "rate_negative_words", "global_rate_positive_words",
            "global_rate_negative_words"]].corr()
dff

sns.pairplot(temp_data[["shares", "rate_positive_words", "rate_negative_words", "global_rate_positive_words",
                        "global_rate_negative_words"]], diag_kind='kde')

X = data[["rate_positive_words", "rate_negative_words", "global_rate_positive_words", "global_rate_negative_words"]]
y = data["shares"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.predict(np.array([[3, 5, 7, 9]]))
y_pred = reg.predict(X_test)
len(y_pred)
y_test
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

data[["rate_positive_words", "rate_negative_words", "global_rate_positive_words", "global_rate_negative_words",
      "shares"]]
reg_pred = reg.predict([[0.769231, 0.230769, 0.045662, 0.013699]])
reg_pred
X1 = data.iloc[:, 2:60]
y1 = data["shares"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X1_train, y1_train)
y1_pred = regressor.predict(X1_test)

mse = round((mean_squared_error(y1_test, y1_pred)) / 100, 2)
rmse = round((sqrt(mse)) / 100, 2)
mse, rmse

X1 = sm.add_constant(X1)
model = sm.OLS(y1, X1).fit()
model.summary()

model.summary()

len(list(X1.columns))

list(X1)

0.023

X1 = X1.drop(['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'num_hrefs', 'data_channel_is_lifestyle',
              'data_channel_is_entertainment', 'data_channel_is_bus', 'average_token_length',
              'data_channel_is_lifestyle', 'data_channel_is_socmed',
              'kw_min_avg', 'kw_avg_avg', 'kw_max_avg', 'global_rate_positive_words', 'global_subjectivity',
              'self_reference_min_shares', 'num_keywords', 'avg_negative_polarity', 'kw_avg_max', 'num_imgs',
              'num_self_hrefs',
              'self_reference_avg_sharess', 'self_reference_avg_sharess', 'n_non_stop_unique_tokens',
              'min_negative_polarity', 'self_reference_max_shares', 'kw_max_max'], axis=1)
len(list(X1.columns))

X1 = sm.add_constant(X1)
model = sm.OLS(y1, X1).fit()
model.summary()

publishdayMerge = data[['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
                        'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']]

temp_arr = []
for r in list(range(publishdayMerge.shape[0])):
    for c in list(range(publishdayMerge.shape[1])):
        if ((c == 0) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Monday')
        elif ((c == 1) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Tueday')
        elif ((c == 2) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Wednesday')
        elif ((c == 3) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Thursday')
        elif ((c == 4) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Friday')
        elif ((c == 5) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Saturday')
        elif ((c == 6) and (publishdayMerge.iloc[r, c]) == 1):
            temp_arr.append('Sunday')

DataChannelMerge = data[['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
                         'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']]

DataChannel_arr = []
for r in list(range(DataChannelMerge.shape[0])):
    if (((DataChannelMerge.iloc[r, 0]) == 0) and ((DataChannelMerge.iloc[r, 1]) == 0) and (
            (DataChannelMerge.iloc[r, 2]) == 0) and ((DataChannelMerge.iloc[r, 3]) == 0) and (
            (DataChannelMerge.iloc[r, 4]) == 0) and ((DataChannelMerge.iloc[r, 5]) == 0)):
        DataChannel_arr.append('Others')
    for c in list(range(DataChannelMerge.shape[1])):
        if ((c == 0) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('Lifestyle')
        elif ((c == 1) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('Entertainment')
        elif ((c == 2) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('Business')
        elif ((c == 3) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('Social Media')
        elif ((c == 4) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('Tech')
        elif ((c == 5) and (DataChannelMerge.iloc[r, c]) == 1):
            DataChannel_arr.append('World')

data.insert(loc=11, column='weekdays', value=temp_arr)
data.insert(loc=12, column='data_channel', value=DataChannel_arr)

data.drop(labels=['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
                  'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',
                  'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
                  'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday'], axis=1,
          inplace=True)
final_data = data.iloc[:, 2:]
final_data

X2 = final_data[['n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'num_hrefs',
                 'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length', 'num_keywords',
                 'kw_avg_avg', 'self_reference_avg_sharess', 'global_subjectivity',
                 'global_sentiment_polarity', 'global_rate_positive_words', 'global_rate_negative_words',
                 'avg_positive_polarity',
                 'avg_negative_polarity', 'title_sentiment_polarity']]

data = pd.read_csv("OnlineNewsPopularity.csv")
data = data.rename(columns=lambda x: x.strip())
data

X = data[["n_tokens_title", "n_tokens_content", "n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens",
          "average_token_length",
          "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech",
          "data_channel_is_world",
          "weekday_is_monday", "weekday_is_tuesday", "weekday_is_wednesday", "weekday_is_thursday", "weekday_is_friday",
          "weekday_is_saturday",
          "weekday_is_sunday", "is_weekend", "global_subjectivity", "title_subjectivity", "abs_title_subjectivity"]]
y = data.iloc[:, 60]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

[i.shape for i in [X_train, X_test, y_train, y_test]]

X_train.info()

y_train.isnull().sum()
X_train.describe()
plt.figure(figsize=(15, 8))
plt.xticks(rotation=45)
sns.boxplot(data=X_train)
print('The scikit-learn version is {}.'.format(sklearn.__version__))
scale = StandardScaler()
Scaled_X_train = pd.DataFrame(scale.fit_transform(X_train), columns=list(X_train.columns))
plt.figure(figsize=(15, 8))
plt.xticks(rotation=45)
sns.boxplot(data=Scaled_X_train)

def cap_outliers(array, k=3):
    upper_limit = array.mean() + k * array.std()
    lower_limit = array.mean() - k * array.std()
    array[array < lower_limit] = lower_limit
    array[array > upper_limit] = upper_limit
    return array

Outlier_X_train = Scaled_X_train.apply(cap_outliers, axis=0)
Outlier_X_train

Scaled_X_train.describe()

Outlier_X_train.describe()

plt.figure(figsize=(15, 8))
plt.xticks(rotation=45)
sns.boxplot(data=Outlier_X_train)
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(Outlier_X_train, y_train.to_numpy().reshape(-1, ))
feature_importances = pd.DataFrame({'col': Outlier_X_train.columns, 'importance': rf.feature_importances_})
plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
plt.bar(feature_importances['col'], feature_importances['importance'])

pca = PCA(n_components=9)
components = pca.fit_transform(Outlier_X_train)
sns.scatterplot(x=components[:, 0], y=components[:, 1])

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(components, y_train.to_numpy().reshape(-1, ))

feature_importances = pd.DataFrame(
    {'col': ['component_' + str(i) for i in range(components.shape[1])], 'importance': rf.feature_importances_})

plt.figure(figsize=(8, 6))
plt.xticks(rotation=90)
plt.bar(feature_importances['col'], feature_importances['importance'])
model = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1)

model.fit(Outlier_X_train, y_train.to_numpy().reshape(-1, ))
y_pred = model.predict(X_test)
len(y_pred)
y_pred
len(y_pred)

mean_squared_error(y_test, y_pred)

scale = StandardScaler()
pca = PCA(n_components=14)
poly = PolynomialFeatures(degree=2)
model = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1)

pipe_lr = Pipeline(steps=[('scaling', scale),
                          ('pca', pca),
                          ('poly', poly),
                          ('model', model)])

def regression_results(y_true, y_pred):

    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))

pipe_lr.fit(X_train, y_train.to_numpy().reshape(-1, ))

train_pred = pipe_lr.predict(X_train)
test_pred = pipe_lr.predict(X_test)

print("Train metrics")
regression_results(y_train.to_numpy().reshape(-1, ), train_pred)

y_pred[y_pred > 50000]

11894

len(list(y_pred[y_pred > 50000]))

y_pred.min()
y_pred.max()
(y_pred.min() + y_pred.max()) / 5
len(y_pred[y_pred < y_pred.min() * 10])
filtered = [num for num in list(y_pred) if y_pred.min() * 10 < num <= y_pred.min() * 15]
len(filtered)
len(y_pred[y_pred > y_pred.min() * 15])
lst_a = [len(y_pred[y_pred < y_pred.min() * 10]), len(filtered), len(y_pred[y_pred > y_pred.min() * 15])]
lst_b = ["Least Popular", "medium Popular", "Popular"]

data_for_plot = pd.DataFrame(list(zip(lst_a, lst_b)), columns=["values", "news"])
data_for_plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=data_for_plot, y="values", x="news")
