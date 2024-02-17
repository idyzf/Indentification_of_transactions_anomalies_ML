import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


data = pd.read_csv("transaction_anomalies_dataset.csv")
print(data.head())
print(data.isnull().sum())
print(data.info())

##########################Visualization###############################
# Distribution of Transactions
hist_amount_graph = px.histogram(data, x='Transaction_Amount',
                          nbins=20,
                          title='Distribution of Transaction Amount')
hist_amount_graph.show()

# Transaction Amount by Account Type
boxplot_amount_graph = px.box(data,
                        x='Account_Type',
                        y='Transaction_Amount',
                      title= "Transaction Amount")

boxplot_amount_graph.show()

#Transaction Amount vs. Age
scatter_age_graph = px.scatter(data, x='Age',
                                        y='Average_Transaction_Amount',
                                        color='Account_Type',
                                        title='Average Transaction Amount vs. Age',
                                        trendline="ols")
scatter_age_graph.show()

#Transactions by Day of the Week
day_of_week = px.bar(data, x='Day_of_Week',
                         title='Count of Transactions by Day of the Week')
day_of_week.show()

# Correlation
df = pd.DataFrame(data)
df_filtered = df.select_dtypes(include=[int, float])
correlation_matrix = df_filtered.corr()
correlation_graph = px.imshow(correlation_matrix,
                             title='Correlation Heatmap')
correlation_graph.show()

############################# Finding Anomalies ####################
# Calculate mean and standard deviation of Transaction Amount
mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()

# Define the anomaly threshold
anomaly_threshold = mean_amount + 2 * std_amount

# Flag anomalies
data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

# Visualization
scatter_anomalies = px.scatter(data, x='Transaction_Amount', y='Average_Transaction_Amount',
                           color='Is_Anomaly', title='Anomalies in Transaction Amount')

scatter_anomalies.update_traces(marker= dict(size=12),
                                selector=dict(mode='markers', marker_size=1))
scatter_anomalies.show()

#Calculating them
num_anomalies = data['Is_Anomaly'].sum()
total_instances = data.shape[0]

# ratio
anomaly_ratio = num_anomalies/total_instances
print(anomaly_ratio)

#####################Training the Model ###################################
#the relevant features will be the ones who had distinctions on the visualization step
features = ['Transaction_Amount',
                     'Average_Transaction_Amount',
                     'Frequency_of_Transactions']

# X -> features and y-> target variable
X = data[features]
y = data['Is_Anomaly']

# Split data into train and test sets
#test_size is the % you want to use as training (the usual is 70-30%)
#random_state is the seed (needs to be the same every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train
#contamination is the ratio
model = IsolationForest (contamination=0.02, random_state=42)
model.fit(X_train)

# Performance of the model
y_pred = model.predict(X_test)

# Convert predictions to binary values (0: normal, 1: anomaly)
y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]

report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'])
print(report)

######## Using the model to predict anomalies######

user_inputs = []
for feature1 in features:
    user_input = float(input(f"Enter the value for '{feature1}': "))
    user_inputs.append(user_input)

user_df = pd.DataFrame([user_inputs], columns=features)
user_anomaly_pred = model.predict(user_df)
user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

if user_anomaly_pred_binary ==1:
    print("Anomaly detected: This transaction is flagged as an anomaly.")
else:
    print("No anomaly detected")