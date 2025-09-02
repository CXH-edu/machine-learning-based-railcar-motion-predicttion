import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.svm import SVR
from itertools import product
from random import choice
from scipy.stats import uniform, randint

np.random.seed(42)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

features = [
    "TotalMass", "TotalLength", "Velocity1", "GravityCenter1",
    "Velocity2", "TheoreticalVelocity2", "GravityCenter2",
    "T12", "TheoreticalT12", "TheoreticalVelocity3", "GravityCenter3"
]
targets = ["Velocity3"]

# 加载数据
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Output")
os.makedirs(output_dir, exist_ok=True)

parent_dir = os.path.dirname(current_dir)
input_file_path = os.path.join(
    parent_dir, "驼峰仿真C#", "Programming", "驼峰仿真美国cars3",
    "驼峰仿真美国cars3", "Output", "2025-06-16_5000车组减速器0-减速器目的.csv"
)
data = pd.read_csv(input_file_path, na_values=["null", "nan", ""])
data = data[data["Velocity3"] > 1]

# 特征与目标归一化
X = data[features]
y = data["Velocity3"]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = pd.DataFrame(scaler_X.fit_transform(X), columns=features)
y = pd.Series(scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(), name="Velocity3_normalized")

# 数据集划分，训练集70%，验证集15%，测试集15%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=42)

# 训练并评估 SVR 模型
def train_svr_and_evaluate(params, X_train, y_train, X_valid, y_valid):
    model = SVR(C=params["C"], epsilon=params["epsilon"], kernel=params["kernel"])
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    return model, valid_rmse

if __name__ == "__main__":
    # 定义随机搜索空间（范围与网格一致）
    param_dist = {
        "C": uniform(loc=0.1, scale=99.9),            # [0.1, 100]
        "epsilon": uniform(loc=0.01, scale=0.09),      # [0.01, 0.1]
        "kernel": ["linear", "rbf"]                    # 离散选取
    }
    n_iterations = 100

    start_time = pd.Timestamp.now()
    print("开始随机搜索:", start_time)

    best_rmse = float("inf")
    best_model = None
    best_params = None
    C_list, epsilon_list, kernel_list, rmse_list = [], [], [], []

    for i in range(n_iterations):
        params = {
            "C": param_dist["C"].rvs(),
            "epsilon": param_dist["epsilon"].rvs(),
            "kernel": choice(param_dist["kernel"])
        }
        model, valid_rmse = train_svr_and_evaluate(params, X_train, y_train, X_valid, y_valid)

        C_list.append(params["C"])
        epsilon_list.append(params["epsilon"])
        kernel_list.append(params["kernel"])
        rmse_list.append(valid_rmse)

        print(f"第{i+1}/{n_iterations}组: {params}, 验证集RMSE: {valid_rmse:.4f}")

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_model = model
            best_params = params
            print(f"更新最佳模型: {params}, RMSE: {valid_rmse:.4f}")

    # 测试集评估
    y_test_pred = best_model.predict(X_test)
    y_test_pred_orig = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
    y_test_orig = scaler_y.inverse_transform(y_test.values.reshape(-1, 1))

    test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_orig, y_test_pred_orig)
    test_mape = np.mean(np.abs((y_test_orig - y_test_pred_orig) / y_test_orig)) * 100
    test_r2 = r2_score(y_test_orig, y_test_pred_orig)

    end_time = pd.Timestamp.now()
    print("\n搜索完成:", end_time)
    print(f"耗时: {(end_time - start_time).total_seconds():.2f}秒")
    print(f"最优参数: {best_params}")
    print(f"测试集 RMSE（原始尺度）: {test_rmse:.4f}")

    # 保存预测值、指标、图像
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

        # === 补充训练集与验证集评估 ===
    y_train_pred = best_model.predict(X_train)
    y_valid_pred = best_model.predict(X_valid)

    y_train_pred_orig = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
    y_valid_pred_orig = scaler_y.inverse_transform(y_valid_pred.reshape(-1, 1))
    y_train_orig = scaler_y.inverse_transform(y_train.values.reshape(-1, 1))
    y_valid_orig = scaler_y.inverse_transform(y_valid.values.reshape(-1, 1))

    train_mse = mean_squared_error(y_train_orig, y_train_pred_orig)
    train_rmse = np.sqrt(train_mse)
    train_mae=mean_absolute_error(y_train_orig,y_train_pred_orig)
    train_mape = np.mean(np.abs((y_train_orig - y_train_pred_orig) / y_train_orig)) * 100
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)

    valid_mse = mean_squared_error(y_valid_orig, y_valid_pred_orig)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae=mean_absolute_error(y_valid_orig,y_valid_pred_orig)
    valid_mape = np.mean(np.abs((y_valid_orig - y_valid_pred_orig) / y_valid_orig)) * 100
    valid_r2 = r2_score(y_valid_orig, y_valid_pred_orig)


    df_metrics = pd.DataFrame({
        "预测目标": [targets[0]],
        "测试集MSE": [test_mse],
        "测试集RMSE": [test_rmse],
        "测试集MAE": [test_mae],
        "测试集MAPE": [test_mape],
        "测试集R2": [test_r2],
        "训练集MSE": [train_mse],
        "训练集RMSE": [train_rmse],
        "训练集MAE":[train_mae],
        "训练集MAPE": [train_mape],
        "训练集R2": [train_r2],
        "验证集MSE": [valid_mse],
        "验证集RMSE": [valid_rmse],
        "验证集MAE":[valid_mae],
        "验证集MAPE": [valid_mape],
        "验证集R2": [valid_r2],
        "C": [best_params["C"]],
        "epsilon": [best_params["epsilon"]],
        "kernel": [best_params["kernel"]],
        "总耗时（秒）": [(end_time - start_time).total_seconds()],
    })
    df_metrics.to_csv(os.path.join(output_dir, f"SVR_RandomSearch_{targets[0]}_Metrics_{current_time}.csv"), index=False, encoding="utf-8-sig")

    df_pred = pd.DataFrame({
        f"实际 {targets[0]}": y_test_orig.flatten(),
        f"预测 {targets[0]}": y_test_pred_orig.flatten()
    })
    df_pred.to_csv(os.path.join(output_dir, f"SVR_RandomSearch_{targets[0]}_Predictions_{current_time}.csv"), index=False, encoding="utf-8-sig")

    df_params = pd.DataFrame({
        "搜索轮次": list(range(1, len(C_list)+1)),
        "C": C_list,
        "epsilon": epsilon_list,
        "kernel": kernel_list,
        "valid_rmse": rmse_list
    })
    df_params.to_csv(os.path.join(output_dir, f"SVR_RandomSearch_{targets[0]}_Parameters_{current_time}.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_orig, y_test_pred_orig, alpha=0.3)
    plt.plot(
        [y_test_orig.min(), y_test_orig.max()],
        [y_test_orig.min(), y_test_orig.max()],
        "r--", lw=2
    )
    plt.xlabel(f"实际 {targets[0]}")
    plt.ylabel(f"预测 {targets[0]}")
    plt.title("SVR 实际值 vs 预测值 (原始尺度)")
    plt.grid(True)
    plt.show()
