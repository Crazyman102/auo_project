from datetime import date
from numpy.lib.type_check import real
from scipy.sparse import data
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import reshape
from numpy.lib.npyio import load
from pandas import read_csv
import joblib
from sklearn.metrics import mean_absolute_error
from tensorflow.python.ops.gen_lookup_ops import lookup_table_export
from numpy import mean

test_data = read_csv('new_dataset/testdata.csv')
real_data = read_csv('new_dataset/realdata.csv')

# load xgboost model
model = joblib.load('model/xgb_model.pkl')
scaler_y = joblib.load('model/scaler_y_noise.pkl')
scaler_X = joblib.load('model/scaler_X_noise.pkl')


# predict
def predict(inputs):
    y_hat = model.predict(test_X)
    y_predict = scaler_y.inverse_transform(y_hat).reshape(7,)
    return y_predict


# plot picture
def plot_picture(dates, predict, actual):
    plt.plot(dates, predict, 'red', label='predict')
    plt.plot(dates, actual, 'green', label='actual')
    plt.title(dates[0]+'~'+dates[-1])
    plt.xticks(rotation=30)
    plt.xlabel('Date')
    plt.ylabel('Energy')
    plt.legend()
    plt.show()

all_mae=[]
check=[]
for i in range(0, 12):
    # MinMaxScaler test_X
    test_X = test_data.iloc[14*i:14*(i+1), 1:].values
    check.append(test_X)
    test_X = scaler_X.fit_transform(test_X)
    test_X = test_X.reshape(1, 84)
   
    # predict label
    outputs=predict(test_X)

    # actual data
    dates = real_data.iloc[7*i:7*(i+1), 0].values.reshape(7,)
    actual_y = real_data.iloc[7*i:7*(i+1), 1:].values.reshape(7,)
    print(actual_y)
    # cal mae
    mae = mean_absolute_error(actual_y, outputs)
    all_mae.append(mae)
    
    # plot_picture
    plot_picture(dates,outputs,actual_y)

