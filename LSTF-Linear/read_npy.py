import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
def rmse_cal(label,prediction):
    mse = np.sum((label - prediction) ** 2) / len(label)
    rmse = np.sqrt(mse)
    return rmse

def make_rmse_list(label,prediction,day_point=1):
    if label.size != prediction.size:
        return -1
    label = label.reshape(-1,1)
    prediction = prediction.reshape(-1,1)
    num = int(288 / day_point)
    rounds = int(label.size / num)
    rmse = []
    rmse1 = 0
    for i in range(rounds):
        y_label = label[i * num:(i+1) * num - 1]
        y_pre = prediction[i * num:(i+1) * num - 1]
        rmse1 = rmse_cal(y_label,y_pre)
        rmse.append(rmse1)
    return rmse

data_name = "5T_gap_1Y"
One_day_rmse = 0.1
data = pd.read_csv("./data/atmosphere_density/"+data_name+".csv",engine='python')


N00 = np.array(data['N00']).reshape(-1,1)
true_den = np.array(data['Grace-A']).reshape(-1,1)

"""目前最佳 216/108/216"""
# ai_calibration
all_pred = np.load("results/calibration/calibration_216.npy")
# correction
# all_pred = np.load("results/all_pred_correction/5T_gap_1Y_test_NDLinear_5T_gap_1Y_ftMS_sl864_ll432_pl864_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy")
# all_pred
all_pred = np.load("results/all_pred/5T_gap_1Y_LastYear_0.25_test_NDLinear_5T_gap_1Y_LastYear_0.25_ftMS_sl216_ll108_pl216_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0/real_prediction.npy")

# metrics = np.load('results/now_testing/metrics.npy')
all_pred = all_pred.reshape(-1,1)
N00 = N00[:all_pred.size]
true_den = true_den[:all_pred.size]

temp = np.concatenate((true_den,all_pred,N00),axis=1)

# 归一化
# scaler = StandardScaler()
# temp = scaler.fit_transform(temp)
# preds = scaler.inverse_transform(preds)

print("N00 rmse = ",rmse_cal(N00,true_den))
print("pre rmse = ",rmse_cal(all_pred,true_den))

print("N00 rmse = ",np.sqrt(np.mean((N00 - true_den) ** 2)))
print("pre rmse = ",np.sqrt(np.mean((all_pred - true_den) ** 2)))

N00_rmse = make_rmse_list(true_den,N00,One_day_rmse)
preds_rmse = make_rmse_list(true_den,all_pred,One_day_rmse)

days = range(0,len(N00_rmse))
days = [i/One_day_rmse for i in days]
plt.figure()
plt.plot(days, N00_rmse, label='N00_rmse', color='g')
plt.plot(days, preds_rmse, label='prediction_rmse', color='r')
plt.xlabel('DAYS')
plt.ylabel('RMSE/ (1e-12*(kg·m^-3))')
plt.legend(loc='upper right')
plt.title("density present")
plt.show()