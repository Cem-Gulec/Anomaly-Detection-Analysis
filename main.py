import datetime
import pandas as pd
import numpy as np
import subprocess
import os
print("This is a program for getting day by day statistics of spesific actions for given time range and channel")

'''

README

This programs drives the log downloading process
You give date interval
You give channel name
You give intensity (how many curls in an hour, accurracy)
Then it creates

directory: /channelname-all-logs-hourly/
temp file: /channelname-all-logs/start-temp.txt
last file: /channelname-all-logs/start-hour.txt



print("Please enter the start date and finish date")


now = datetime.datetime.now()

start_year = now.year
start_month = now.month
start_day = now.day
start_hour = now.hour

channel_name="WebCredit"

intensity = 12

directory = channel_name+"-all-logs-hourly"
mins = 60/intensity

start = datetime.datetime(int(start_year),int(start_month),int(start_day),start_hour)
start2 = start
end_date = start + datetime.timedelta(minutes=60)



for z in range(12):

        end = start + datetime.timedelta(minutes=5) # Do the job between this 2


        from_date = str(start.year)+'-'+str(start.month)+'-'+str(start.day) # date for from part of script
        to_date = str(end.year)+'-'+str(end.month)+'-'+str(end.day)  # date for to part of script

        from_hour = str(start.hour)        # hour for from part
        to_hour = str(end.hour)            # hour for to part

        from_minutes = str(start.minute)  # minutes for from part
        to_minutes = str(end.minute)      # minutes for end part



        if(start.hour<10):               # string formatting
            from_hour="0"+from_hour      # string formatting
        if(end.hour<10):                 # string formatting
            to_hour="0"+to_hour          # string formatting

        if(start.minute<10):                  # string formatting
            from_minutes="0"+from_minutes     # string formatting
        if(end.minute<10):                    # string formatting
            to_minutes="0"+to_minutes         # string formatting

        temp_file = from_date+"-temp.txt"
        last_file = from_date+"-"+from_hour+".txt"

        curl_sc = "#!/bin/sh\ncurl -u taha.komur:Taha123! -H 'Accept:application/json' -X GET \"http://10.90.11.90:9000/api/search/universal/absolute?query=ChannelName%3A"+channel_name+"&from="+from_date+"%20"+from_hour+"%3A"+from_minutes+"%3A00.000&to="+to_date+"%20"+to_hour+"%3A"+to_minutes+"%3A00.000&limit=10000&fields=Action&decorate=true\" >"+temp_file


        file = open("curl_temp.sh","w+")
        file.write(curl_sc)
        file.close()
        os.chmod("curl_temp.sh", 0o777)
        subprocess.call("./curl_temp.sh")


        grep_sc = "#!/bin/sh\ncat "+temp_file+" | jq '.messages[].message' | grep \"Action\"  >> "+last_file+" | rm "+temp_file+" -f"
        file = open("grep_temp.sh","w+")
        file.write(grep_sc)
        file.close()
        #print("Curling: ",curl_sc)
        os.chmod("grep_temp.sh",0o777)
        subprocess.call("./grep_temp.sh")

        #print(curl_sc+"\n")
        #print(grep_sc+"\n")
        #print(from_date,' ',from_hour,':',from_minutes,"    ",to_date,' ',to_hour,':',to_minutes)
        start += datetime.timedelta(minutes=5)     # Do the job between this 2
        if(start == end_date):
            break

string = str(start2.year)+'-'+str(start2.month)+'-'+str(start2.day)+' '+str(start2.hour)+':00:00,'
append_sc = "#!/bin/sh\necho -n "+string+" >> ApplyCreditApplication-statistics.txt | cat "+last_file+" | grep \"ApplyCreditApplication\" | wc -l >> ApplyCreditApplication-statistics.txt"
file = open("append_sc.sh","w+")
file.write(append_sc)
file.close()
#print("Curling: ",curl_sc)
os.chmod("append_sc.sh",0o777)
subprocess.call("./append_sc.sh")
'''
from fbprophet import Prophet
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pyemma import msm
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import sys
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.layers import Input, Dense,Activation, Dropout
import warnings


class K_means():

    def __init__(self):
        self.n_clusters = 7

    # return Series of distance between each point and his distance with the closest centroid
    def getDistanceByPoint(self, data, model):
        distance = pd.Series(dtype='float64')
        cluster_cnt = model.labels_.max()
        for i in range(0, len(data)):
            Xa = np.array(data.loc[i])
            dist = []
            for j in range(0, cluster_cnt):
                if model.labels_[i] != j:
                    Xb = model.cluster_centers_[j]
                    dist.append(np.linalg.norm(Xa - Xb))
            distance.at[i] = min(dist)
        return distance

    # Plot k-means clusters by different colors
    def cluster_colored_visual(self):

        # plot the different clusters with the 2 main features
        fig, ax = plt.subplots()
        colors = {0:'red', 1:'blue', 2:'green', 3:'pink', 4:'black', 5:'orange', 6:'cyan', 7:'yellow', 8:'brown', 9:'purple'}
        ax.scatter(df['principal_feature1'], df['principal_feature2'], c=df["cluster"].apply(lambda x: colors[x]))

        plt.show()
    
    # For selecting best possible k value
    def anomalyDetection(self):
        # Take useful feature and standardize them
        data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # reduce to 2 importants features
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)

        # standardize these 2 new features
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # calculate with different number of centroids to see the loss plot (elbow method)
        # In theory most useful one is selected depending on where the curvature goes near zero
        n_cluster = range(1, 9)
        kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]

        # I choose 6 centroids arbitrarily and add these data to the central dataframe
        df['cluster'] = kmeans[self.n_clusters].predict(data)
        df['principal_feature1'] = data[0]
        df['principal_feature2'] = data[1]

        # get the distance between each point and its nearest centroid. 
        # The biggest distances are considered as anomaly
        distance = self.getDistanceByPoint(data, kmeans[self.n_clusters])
        number_of_outliers = int(outliers_fraction*len(distance))

        # Take the minimum of the largest x% of the distances as the threshold
        threshold = distance.nlargest(number_of_outliers).min()

        # anomaly21 contain the anomaly result of method 2.1 Cluster (0:normal, 1:anomaly) 
        df['anomaly21'] = (distance >= threshold).astype(int)

        timestamp_dict = {}

        anomaly21_df = df[df['anomaly21'] == 1].hour
            
        for index, value in anomaly21_df.items():
            timestamp_dict[value] = 1

        self.cluster_colored_visual()
        
        return timestamp_dict
    
class Elliptic_Envelope():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict
    
    def anomalyDetection(self):

        # creation of 4 differents data set based on categories defined before
        df_class0 = df.loc[df['categories'] == 0, 'value']
        df_class1 = df.loc[df['categories'] == 1, 'value']
        df_class2 = df.loc[df['categories'] == 2, 'value']
        df_class3 = df.loc[df['categories'] == 3, 'value']

        # apply ellipticEnvelope(gaussian distribution) at each categories
        envelope =  EllipticEnvelope(contamination = outliers_fraction) 
        X_train = df_class0.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class0 = pd.DataFrame(df_class0)
        df_class0['deviation'] = envelope.decision_function(X_train)
        df_class0['anomaly'] = envelope.predict(X_train)

        envelope =  EllipticEnvelope(contamination = outliers_fraction) 
        X_train = df_class1.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class1 = pd.DataFrame(df_class1)
        df_class1['deviation'] = envelope.decision_function(X_train)
        df_class1['anomaly'] = envelope.predict(X_train)

        envelope =  EllipticEnvelope(contamination = outliers_fraction) 
        X_train = df_class2.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class2 = pd.DataFrame(df_class2)
        df_class2['deviation'] = envelope.decision_function(X_train)
        df_class2['anomaly'] = envelope.predict(X_train)

        envelope =  EllipticEnvelope(contamination = outliers_fraction) 
        X_train = df_class3.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class3 = pd.DataFrame(df_class3)
        df_class3['deviation'] = envelope.decision_function(X_train)
        df_class3['anomaly'] = envelope.predict(X_train)

        # add the data to the main 
        df_class = pd.concat([df_class0, df_class1, df_class2, df_class3])
        df['anomaly22'] = df_class['anomaly']
        df['anomaly22'] = np.array(df['anomaly22'] == -1).astype(int) 

        anomaly22_df = df[df['anomaly22'] == 1].hour
            
        for index, value in anomaly22_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

class Prophet_Forecast():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict

    def fit_predict_model(self, dataframe, interval_width = 0.99, changepoint_range = 0.8):
    
        m = Prophet(daily_seasonality = False, yearly_seasonality = False,
                weekly_seasonality = False, seasonality_mode = 'multiplicative',
                interval_width = interval_width, changepoint_range = changepoint_range)
        m = m.fit(dataframe)
        
        forecast = m.predict(dataframe)
        forecast['fact'] = dataframe['y'].reset_index(drop = True)
                
        return forecast
    
    def detect_anomalies(self, forecast):
    
        forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
        
        forecasted['anomaly'] = 0
        forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
        forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1
        
        # anomaly importances
        forecasted['importance'] = 0
        forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = (forecasted['fact'] - forecasted['yhat_upper']) / forecast['fact']
        forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = (forecasted['yhat_lower'] - forecasted['fact']) / forecast['fact']
        
        return forecasted
    
    def anomalyDetection(self):
        
        pred = self.fit_predict_model(df_forecast)
        pred = self.detect_anomalies(pred)

        # Adding anomalies detected from prophet algorithm as column to dataframe
        df['anomaly23'] = pred['anomaly']

        anomaly23_df = pred[pred['anomaly'] == 1].ds
            
        for index, value in anomaly23_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

class Markov_Chains():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict

    # train markov model to get transition matrix
    def getTransitionMatrix(self,df):
        df = np.array(df)
        model = msm.estimate_markov_model(df, 1)
        return model.transition_matrix

    def markovAnomaly(self, df, windows_size, threshold):
        transition_matrix = self.getTransitionMatrix(df)
        real_threshold = threshold**windows_size
        df_anomaly = []
        for j in range(0, len(df)):
            if (j < windows_size):
                df_anomaly.append(0)
            else:
                sequence = df[j-windows_size:j]
                sequence = sequence.reset_index(drop=True)
                df_anomaly.append(self.anomalyElement(sequence, real_threshold, transition_matrix))
        return df_anomaly

    def successProbabilityMetric(self, state1, state2, transition_matrix):
        proba = 0
        for k in range(0,len(transition_matrix)):
            if (k != (state2-1)):
                proba += transition_matrix[state1-1][k]
        return 1-proba

    def sucessScore(self, sequence, transition_matrix):
        proba = 0
        for i in range(1,len(sequence)):
            if(i == 1):
                proba = self.successProbabilityMetric(sequence[i-1], sequence[i], transition_matrix)
            else:
                proba = proba*self.successProbabilityMetric(sequence[i-1], sequence[i], transition_matrix)
        return proba

    def anomalyElement(self, sequence, threshold, transition_matrix):
        if (self.sucessScore(sequence, transition_matrix) > threshold):
            return 0
        else:
            return 1

    def anomalyDetection(self):

        # definition of the different state
        x1 = (df['value'] <= 17).astype(int)
        x2 = ((df['value'] > 17) & (df['value'] <= 30)).astype(int)
        x3 = ((df['value'] > 30) & (df['value'] <= 43)).astype(int)
        x4 = ((df['value'] > 43) & (df['value'] <= 100)).astype(int)
        x5 = (df['value'] > 100).astype(int)
        df_mm = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5

        # getting the anomaly labels for our dataset (evaluating sequence of 5 values and anomaly = less than 20% probable)
        df_anomaly = self.markovAnomaly(df_mm, 5, 0.20)
        df_anomaly = pd.Series(df_anomaly)

        # add the data to the main 
        df['anomaly24'] = df_anomaly

        anomaly24_df = df[df['anomaly24'] == 1].hour
      
        for index, value in anomaly24_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

class Isolation_Forest():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict

    def anomalyDetection(self):
        
        # Take useful feature and standardize them 
        data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        # train isolation forest 
        model =  IsolationForest(contamination = outliers_fraction)
        model.fit(data)

        # add the data to the main  
        df['anomaly25'] = pd.Series(model.predict(data))
        df['anomaly25'] = df['anomaly25'].map( {1: 0, -1: 1} )

        anomaly25_df = df[df['anomaly25'] == 1].hour
      
        for index, value in anomaly25_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

class SVM():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict
    
    def anomalyDetection(self):
        # Take useful feature and standardize them 
        data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)

        # train one class SVM 
        model =  OneClassSVM(nu=0.95 * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
        data = pd.DataFrame(np_scaled)
        model.fit(data)

        # add the data to the main  
        df['anomaly26'] = pd.Series(model.predict(data))
        df['anomaly26'] = df['anomaly26'].map( {1: 0, -1: 1} )

        anomaly26_df = df[df['anomaly26'] == 1].hour
      
        for index, value in anomaly26_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

class LSTM():

    def __init__(self, timestamp_dict):
        self.timestamp_dict = timestamp_dict

    # unroll: create sequence of 50 previous data points for each data points
    def unroll(self, data,sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)
    
    def anomalyDetection(self):
        # select and standardize data
        data_n = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data_n)
        data_n = pd.DataFrame(np_scaled)

        # important parameters and train/test size
        prediction_time = 1 
        testdatasize = 1000
        unroll_length = 50
        testdatacut = testdatasize + unroll_length  + 1

        #train data
        x_train = data_n[0:-prediction_time-testdatacut].to_numpy()
        y_train = data_n[prediction_time:-testdatacut  ][0].to_numpy()

        # test data
        x_test = data_n[0-testdatacut:-prediction_time].to_numpy()
        y_test = data_n[prediction_time-testdatacut:  ][0].to_numpy()

        # adapt the datasets for the sequence data shape
        x_train = self.unroll(x_train,unroll_length)
        x_test  = self.unroll(x_test,unroll_length)
        y_train = y_train[-x_train.shape[0]:]
        y_test  = y_test[-x_test.shape[0]:]

        # Build the model
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(x_train.shape[1], 5), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='rmsprop')
        #print(model.summary())

        # Train the model
        history = model.fit( 
            x_train,
            y_train,
            batch_size = 3028,
            epochs = 30,
            validation_split = 0.1)

        # create the list of difference between prediction and test data
        loaded_model = model
        diff=[]
        ratio=[]
        p = loaded_model.predict(x_test)

        # predictions = lstm.predict_sequences_multiple(loaded_model, x_test, 50, 50)
        for u in range(len(y_test)):
            pr = p[u][0]
            ratio.append((y_test[u]/pr)-1)
            diff.append(abs(y_test[u]- pr))

        # select the most distant prediction/reality data points as anomalies
        diff = pd.Series(diff)
        number_of_outliers = int(outliers_fraction*len(diff))
        threshold = diff.nlargest(number_of_outliers).min()

        # data with anomaly label (test data part)
        test = (diff >= threshold).astype(int)

        # the training data part where we didn't predict anything (overfitting possible): no anomaly
        complement = pd.Series(0, index=np.arange(len(data_n)-testdatasize))

        # add the data to the main
        df['anomaly27'] = complement.append(test, ignore_index='True')

        anomaly27_df = df[df['anomaly27'] == 1].hour
      
        for index, value in anomaly27_df.items():
            if value in self.timestamp_dict:
                self.timestamp_dict[value] += 1
            else:
                self.timestamp_dict[value] = 1

        return self.timestamp_dict

def plot_histogram(df,name):
        ######## ######## ######## #######
    # Filter dataframe depending on the type selected
    
    str1 = str(df['hour'].iloc[0])
    str2 = str(df['hour'].iloc[-1])
    
    df.plot(x='hour', y='value')
    plt.title(str1 + '  to  ' + str2)

    # Save file depending on the type selected
    plt.savefig('./'+name)
    
    ####### ####### ####### #######

def risk_histogram(dataframe, df_merged, name):

    fig, ax = plt.subplots()
    
    ax.grid(True, linestyle='dotted')
    
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=(mdates.MO)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gcf().autofmt_xdate()
    
    ax.plot(dataframe['hour'], dataframe['value'], color='coral', zorder=1)
    ax.scatter(df_merged['hour'], df_merged['value'], color='blue', zorder=2)

    commonBy_list = []

    for index, row in df_merged.iterrows():
        commonBy_list.append(row['CommonBy'])

    for i, txt in enumerate(commonBy_list):
        risk_value = (int(txt) / 6) * 100
        txt = str(float("{:.2f}".format(risk_value))) + "%"
        ax.annotate(str(txt), (df_merged['hour'].iloc[i], df_merged['value'].iloc[i] + 5), color = "black", weight = "bold", zorder=3) 
    
    # Save file depending on the type selected
    plt.savefig('./' + name)

warnings.simplefilter(action='ignore', category=FutureWarning)

normal_file_path = "output.txt"
forecast_file_path = "output_forecast.txt"


df = pd.read_csv(normal_file_path)
df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values(by="hour")

######## ######## ######## #######

from matplotlib import pyplot as plt
import pandas as pd


# Filter dataframe depending on the type selected

df_daily = df.iloc[-24*1:]
df_weekly = df.iloc[-24*7:]
df_monthly = df.iloc[-24*30:]
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100 
plot_histogram(df_daily,'daily')
plot_histogram(df_weekly,'weekly')
plot_histogram(df_monthly,'monthly')
plot_histogram(df,'all')

####### ####### ####### #######

df_forecast = pd.read_csv(forecast_file_path)
df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
df_forecast = df_forecast.sort_values(by="ds")

# change the type of hour column for plotting
# timestamp to hours
df['hours'] = df['hour'].dt.hour
# declaring daylight on hours of 7,..,17
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 19)).astype(int)
# Monday=0,.., Sunday=6
df['DayOfTheWeek'] = df['hour'].dt.dayofweek
df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
# time with int to plot easily
df['time_epoch'] = (df['hour'].astype(np.int64)/100000000000).astype(np.int64)
# creation of 4 distinct categories that seem useful (week end/day week & night/day)
df['categories'] = df['WeekDay']*2 + df['daylight']
# An estimation of anomly population of the dataset 
outliers_fraction = 0.01


# K-means clustering implementation
# return type: dict
K_means_anomalies = K_means().anomalyDetection()

# Gaussian + Elliptic Envelope implementation
# return type: dict
#EllipticEnvelope_anomalies = Elliptic_Envelope(K_means_anomalies).anomalyDetection()

# Prophet Forecasting implementation
#Forecast_anomalies = Prophet_Forecast(EllipticEnvelope_anomalies).anomalyDetection()

# Markov Chains implementation
#Markov_Chains_anomalies = Markov_Chains(Forecast_anomalies).anomalyDetection()

# Isolation Forest implementation
#Isolation_Forest_anomalies = Isolation_Forest(Markov_Chains_anomalies).anomalyDetection()

# Support Vector Machine (SVM) implementation
#SVM_anomalies = SVM(Isolation_Forest_anomalies).anomalyDetection()

# Long short-term memory (LSTM) implementation
#LSTM_anomalies = LSTM(SVM_anomalies).anomalyDetection()
match_flag = False
match_value = str(df['hour'].iloc[-1])

### FILE PART ###

data_frame = pd.DataFrame(K_means_anomalies.items(), columns=['hour', 'CommonBy'])
data_frame['hour'] = pd.to_datetime(data_frame['hour'])
data_frame = data_frame.sort_values(by="hour")

data_frame = data_frame[data_frame['CommonBy'] > 1]  
now = datetime.datetime.now()
last_day = now - datetime.timedelta(days=1)
last_week = now - datetime.timedelta(days=7)
last_month = now - datetime.timedelta(days=30)

mask1 = (data_frame['hour'] > last_day)
mask2 = (data_frame['hour'] > last_week)
mask3 = (data_frame['hour'] > last_month)

df_last_day=data_frame.loc[mask1]
df_last_week=data_frame.loc[mask2]
df_last_month=data_frame.loc[mask3]
df_all = data_frame

df_merged_day =   df_last_day.merge(df, how='inner', on='hour')
df_merged_week =  df_last_week.merge(df, how='inner', on='hour')
df_merged_month = df_last_month.merge(df, how='inner', on='hour')
df_merged_all =   df_all.merge(df, how='inner', on='hour')

#risk_histogram(df_daily, df_merged_day, "daily")
#risk_histogram(df_weekly ,df_merged_week, "weekly")
#risk_histogram(df_monthly, df_merged_month, "monthly")
risk_histogram(df, df_merged_all, "all")

######## INFO PAGE PART ########





######## FILE PART ########


for w in sorted(K_means_anomalies, key=K_means_anomalies.get, reverse=True):
    if K_means_anomalies[w] == 1:
        break
    #print(w, SVM_anomalies[w])
    
    if w.strftime('%Y-%m-%d %H:%M:%S')  == match_value:
        match_flag = True
        break

print(match_flag) 