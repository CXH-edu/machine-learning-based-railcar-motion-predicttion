import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from itertools import product
from catboost import CatBoostRegressor, Pool

# 设置随机种子以确保可重复性
np.random.seed(42)

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows: 使用 SimHei 字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方框的问题

features = [
    "TotalMass",
    "TotalLength",
    "Velocity1",
    "GravityCenter1",
    "Velocity2",
    "TheoreticalVelocity2",
    "GravityCenter2",
    "T12",
    "TheoreticalT12",
    "TheoreticalVelocity3",
    "GravityCenter3",
]
targets = ["Velocity3"]

# 加载数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "Output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建 Output 文件夹：{output_dir}")

parent_dir = os.path.dirname(current_dir)
print("当前文件夹所在的根目录:", parent_dir)

input_file_path = os.path.join(
    parent_dir,
    "驼峰仿真C#",
    "Programming",
    "驼峰仿真美国cars3",
    "驼峰仿真美国cars3",
    "Output",
    "2025-06-16_5000车组减速器0-减速器目的.csv",
)
print("目标文件路径:", input_file_path)
data = pd.read_csv(input_file_path, na_values=["null", "nan", ""])

# 筛选 Velocity3 > 1 的数据
data = data[data["Velocity3"] > 1]
print(f"筛选后数据量（Velocity3 > 1）：{len(data)} 条")

# 分离特征和目标变量
X = data[features]
y = data["Velocity3"]

# 初始化MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 对特征 X 进行归一化
X_normalized = scaler_X.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=features)

# 对目标变量 y 进行归一化
y_normalized = scaler_y.fit_transform(y.values.reshape(-1, 1))
y_normalized = pd.Series(y_normalized.flatten(), name="Velocity3_normalized")

print("\n归一化后 X 的前5行:\n", X_normalized.head())
print("\n归一化后 y 的前5行:\n", y_normalized.head())

X = X_normalized
y = y_normalized

# 划分数据集，训练集70%，验证集15%，测试集15%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=42
)

# 自定义CatBoost训练函数以实现带阈值的早停
def train_catboost_with_threshold(
    params,
    X_train,
    y_train,
    X_valid,
    y_valid,
    max_iter=1000,
    patience=10,
    min_delta=1e-4,
):
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)

    params = {
        "learning_rate": params["learning_rate"],
        "depth": params["depth"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "subsample": params["subsample"],
        "loss_function": "RMSE",
        "random_seed": 42,
        "bagging_temperature": 0,
        "random_strength": 1,
        "verbose": 0,  # 禁用日志输出
    }

    current_model = None
    best_valid_loss = float("inf")
    no_improve_count = 0
    early_stop_iter = max_iter
    train_rmse_history = []
    valid_rmse_history = []

    for i in range(1, max_iter + 1):
        temp_model = CatBoostRegressor(
            iterations=1,
            **params,
        )
        temp_model.fit(train_pool, verbose=False, init_model=current_model)
        current_model = temp_model

        train_pred = current_model.predict(X_train)
        valid_pred = current_model.predict(X_valid)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
        train_rmse_history.append(train_rmse)
        valid_rmse_history.append(valid_rmse)

        if best_valid_loss - valid_rmse > min_delta:
            best_valid_loss = valid_rmse
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                early_stop_iter = i
                break

    return current_model, early_stop_iter, train_rmse_history, valid_rmse_history


if __name__ == "__main__":
    # 定义优化的超参数网格
    param_grid = {
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
    }

    start_time = pd.Timestamp.now()
    print("开始进行网格超参数搜索..., 开始时间:", start_time)

    # 记录搜索结果
    learning_rates = []
    depths = []
    l2_leaf_regs = []
    subsamples = []
    valid_rmses = []
    early_stop_iters = []
    best_train_rmse_history = []
    best_valid_rmse_history = []

    # 手动网格搜索，结合带阈值的早停
    best_rmse = float("inf")
    best_params = None
    best_model = None
    param_combinations = list(product(*param_grid.values()))
    n_iterations = len(param_combinations)
    print(f"总共 {n_iterations} 种超参数组合")

    for i, param_values in enumerate(param_combinations):
        iteration_start_time = pd.Timestamp.now()
        params = {
            "learning_rate": param_values[0],
            "depth": param_values[1],
            "l2_leaf_reg": param_values[2],
            "subsample": param_values[3],
        }
        learning_rates.append(params["learning_rate"])
        depths.append(params["depth"])
        l2_leaf_regs.append(params["l2_leaf_reg"])
        subsamples.append(params["subsample"])

        # 训练模型
        model, early_stop_iter, train_rmse_history, valid_rmse_history = (
            train_catboost_with_threshold(
                params,
                X_train,
                y_train,
                X_valid,
                y_valid,
                max_iter=1000,
                patience=10,
                min_delta=1e-4,
            )
        )

        # 在验证集上评估
        y_valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        valid_rmses.append(valid_rmse)
        early_stop_iters.append(early_stop_iter)

        iteration_end_time = pd.Timestamp.now()
        iteration_time_span = iteration_end_time - iteration_start_time
        print(
            f"迭代 {i+1}/{n_iterations}, 耗时：{iteration_time_span.total_seconds():.4f} s, RMSE: {valid_rmse:.4f}\n 超参数组合为：{params}"
        )

        # 更新最佳模型
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_params = params
            best_model = model
            best_train_rmse_history = train_rmse_history
            best_valid_rmse_history = valid_rmse_history
            print(f"新最佳 RMSE: {best_rmse:.4f}, 参数: {best_params}")

    end_time = pd.Timestamp.now()
    time_span = end_time - start_time
    print("\n网格超参数搜索完成, 结束时间:", end_time)
    print(f"总耗时: {time_span.total_seconds():.2f} 秒")
    print("最佳参数:", best_params)
    print(f"最佳RMSE (归一化后): {best_rmse:.4f}")
    print(f"最佳模型在 {early_stop_iters[-1]} 迭代处早停。")

    # 在测试集上进行预测
    y_pred_normalized = best_model.predict(X_test)

    # 将预测结果和真实值还原到原始尺度
    y_pred_original = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1))

    # 确定输出目录
    output_dir = os.path.join(current_dir, "Output")
    current_time = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    # 评估最终模型
    test_mse = mean_squared_error(y_test_original, y_pred_original)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_original, y_pred_original)
    test_mape = (
        np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    )
    test_r2 = r2_score(y_test_original, y_pred_original)

    print(f"\n最终模型在原始尺度测试集上的评估结果:")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"R2 Score: {test_r2:.4f}")

    y_pred_train_original = scaler_y.inverse_transform(
        best_model.predict(X_train).reshape(-1, 1)
    )
    y_train_original = scaler_y.inverse_transform(y_train.values.reshape(-1, 1))

    train_mse = mean_squared_error(y_train_original, y_pred_train_original)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_original, y_pred_train_original)
    train_mape = (
        np.mean(np.abs((y_train_original - y_pred_train_original) / y_train_original))
        * 100
    )
    train_r2 = r2_score(y_train_original, y_pred_train_original)

    y_pred_valid_original = scaler_y.inverse_transform(
        best_model.predict(X_valid).reshape(-1, 1)
    )
    y_valid_original = scaler_y.inverse_transform(y_valid.values.reshape(-1, 1))

    valid_mse = mean_squared_error(y_valid_original, y_pred_valid_original)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = mean_absolute_error(y_valid_original, y_pred_valid_original)
    valid_mape = (
        np.mean(np.abs((y_valid_original - y_pred_valid_original) / y_valid_original))
        * 100
    )
    valid_r2 = r2_score(y_valid_original, y_pred_valid_original)

    df = pd.DataFrame(
        {
            "预测目标": [targets[0]],
            "测试集MSE": [test_mse],
            "测试集RMSE": [test_rmse],
            "测试集MAE": [test_mae],
            "测试集MAPE": [test_mape],
            "测试集R2": [test_r2],
            "训练集MSE": [train_mse],
            "训练集RMSE": [train_rmse],
            "训练集MAE": [train_mae],
            "训练集MAPE": [train_mape],
            "训练集R2": [train_r2],
            "验证集MSE": [valid_mse],
            "验证集RMSE": [valid_rmse],
            "验证集MAE": [valid_mae],
            "验证集MAPE": [valid_mape],
            "验证集R2": [valid_r2],
            "learning_rate": [best_params["learning_rate"]],
            "depth": [best_params["depth"]],
            "l2_leaf_reg": [best_params["l2_leaf_reg"]],
            "subsample": [best_params["subsample"]],
            "early_stop_iter": [early_stop_iters[-1]],
            "总耗时（秒）": [time_span.total_seconds()],
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_{targets[0]}_Metrics_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_file_path}")

    # 绘制实际值 vs. 预测值图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.3)
    plt.plot(
        [y_test_original.min(), y_test_original.max()],
        [y_test_original.min(), y_test_original.max()],
        "r--",
        lw=2,
    )
    plt.xlabel(f"实际 {targets[0]}")
    plt.ylabel(f"预测 {targets[0]}")
    plt.title("CatBoost 实际值 vs. 预测值 (原始尺度)")
    plt.grid(True)
    plt.show()

    df = pd.DataFrame(
        {
            f"实际 {targets[0]}": y_test_original.flatten(),
            f"预测 {targets[0]}": y_pred_original.flatten(),
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_{targets[0]}_Predictions_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存到: {output_file_path}")

    # 绘制最佳模型的损失变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(best_train_rmse_history)),
        best_train_rmse_history,
        label="训练集 RMSE",
    )
    plt.plot(
        range(len(best_valid_rmse_history)),
        best_valid_rmse_history,
        label="验证集 RMSE",
    )
    plt.xlabel("迭代次数")
    plt.ylabel("RMSE")
    plt.title("CatBoost 最优模型的训练和验证集 RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    df = pd.DataFrame(
        {
            "迭代次数": range(len(best_train_rmse_history)),
            "训练集 RMSE": best_train_rmse_history,
            "验证集 RMSE": best_valid_rmse_history,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_{targets[0]}_RMSE_History_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\nRMSE 历史记录已保存到: {output_file_path}")

    df = pd.DataFrame(
        {
            "搜索轮次": range(len(learning_rates)),
            "learning_rate": learning_rates,
            "depth": depths,
            "l2_leaf_reg": l2_leaf_regs,
            "subsample": subsamples,
            "early_stop_iter": early_stop_iters,
            "valid_rmse": valid_rmses,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_{targets[0]}_Parameters_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n超参数记录已保存到: {output_file_path}")
