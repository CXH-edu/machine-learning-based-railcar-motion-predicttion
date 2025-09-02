import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from itertools import product  # 导入 itertools.product 用于网格搜索

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
    "TheoreticalT23",
    "GravityCenter3",
]
targets = ["T23"]

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
y = data["T23"]

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
# 0.17647=0.15/(1-0.15)，确保验证集占总数据的15%


# 自定义 CatBoost 回归器类
class CatBoostRegressorWithEarlyStopping(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate=0.03,
        depth=6,
        min_data_in_leaf=1,
        l2_leaf_reg=3,
        max_iter=1000,
        patience=10,
        min_delta=1e-4,
        random_seed=42,
    ):
        self.learning_rate = learning_rate
        self.depth = depth
        self.min_data_in_leaf = min_data_in_leaf
        self.l2_leaf_reg = l2_leaf_reg
        self.max_iter = max_iter
        self.patience = patience
        self.min_delta = min_delta
        self.random_seed = random_seed
        self.model_ = None
        self.early_stop_iter_ = None
        self.train_rmse_history_ = []  # 记录训练集 RMSE
        self.valid_rmse_history_ = []  # 记录验证集 RMSE

    def fit(self, X, y, eval_set=None):
        if eval_set is None:
            X_train_fit, X_valid_fit, y_train_fit, y_valid_fit = train_test_split(
                X, y, test_size=0.1, random_state=self.random_seed
            )
        else:
            X_train_fit, y_train_fit = X, y
            X_valid_fit, y_valid_fit = eval_set

        train_pool = Pool(X_train_fit, y_train_fit)
        valid_pool = Pool(X_valid_fit, y_valid_fit)

        current_model = None
        no_improve_count = 0
        best_valid_loss = float("inf")
        self.early_stop_iter_ = self.max_iter
        self.train_rmse_history_ = []
        self.valid_rmse_history_ = []

        for i in range(1, self.max_iter + 1):
            temp_model = CatBoostRegressor(
                iterations=1,
                learning_rate=self.learning_rate,
                depth=self.depth,
                min_data_in_leaf=self.min_data_in_leaf,
                l2_leaf_reg=self.l2_leaf_reg,
                loss_function="RMSE",
                verbose=False,
                random_seed=self.random_seed,
            )
            temp_model.fit(train_pool, init_model=current_model)
            current_model = temp_model

            # 计算训练集和验证集的 RMSE
            train_rmse = current_model.eval_metrics(train_pool, ["RMSE"])["RMSE"][-1]
            valid_rmse = current_model.eval_metrics(valid_pool, ["RMSE"])["RMSE"][-1]
            self.train_rmse_history_.append(train_rmse)
            self.valid_rmse_history_.append(valid_rmse)

            if best_valid_loss - valid_rmse > self.min_delta:
                best_valid_loss = valid_rmse
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    self.early_stop_iter_ = i
                    break

        self.model_ = current_model
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError(
                "This CatBoostRegressorWithEarlyStopping instance is not fitted yet. Call 'fit' before using 'predict'."
            )
        return self.model_.predict(X)


if __name__ == "__main__":
    # 定义超参数网格
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7],
        "min_data_in_leaf": [2, 5, 10],
        "l2_leaf_reg": [1, 3, 5]
    }

    # 计算网格搜索的总迭代次数
    n_iterations = np.prod([len(param_grid[key]) for key in param_grid])

    # 自定义超参数搜索
    best_rmse = float("inf")
    best_params = None
    best_model = None

    learning_rates = []
    depths = []
    min_data_in_leafs = []
    l2_leaf_regs = []
    early_stop_iters = []
    valid_rmses = []

    start_time = pd.Timestamp.now()
    print("开始进行网格搜索..., 开始时间:", start_time)

    # 使用 itertools.product 生成所有超参数组合
    param_combinations = list(product(
        param_grid["learning_rate"],
        param_grid["depth"],
        param_grid["min_data_in_leaf"],
        param_grid["l2_leaf_reg"]
    ))

    for i, (learning_rate, depth, min_data_in_leaf, l2_leaf_reg) in enumerate(param_combinations):
        # 构造当前超参数组合
        params = {
            "learning_rate": learning_rate,
            "depth": depth,
            "min_data_in_leaf": min_data_in_leaf,
            "l2_leaf_reg": l2_leaf_reg
        }
        learning_rates.append(params["learning_rate"])
        depths.append(params["depth"])
        min_data_in_leafs.append(params["min_data_in_leaf"])
        l2_leaf_regs.append(params["l2_leaf_reg"])

        iteration_start_time = pd.Timestamp.now()

        # 训练模型
        model = CatBoostRegressorWithEarlyStopping(
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            min_data_in_leaf=params["min_data_in_leaf"],
            l2_leaf_reg=params["l2_leaf_reg"],
            max_iter=1000,
            patience=10,
            min_delta=1e-4,
            random_seed=42,
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        # 在验证集上评估
        y_valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        valid_rmses.append(valid_rmse)

        iteration_end_time = pd.Timestamp.now()
        iteration_time_span = iteration_end_time - iteration_start_time
        print(f"迭代 {i+1}/{n_iterations},耗时: {iteration_time_span.total_seconds():.2f} 秒")

        # 更新最佳模型
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_params = params
            best_model = model
            print(
                f"迭代 {i+1}/{n_iterations}, 新最佳 RMSE: {best_rmse:.4f}, 参数: {best_params}"
            )

        early_stop_iters.append(model.early_stop_iter_)

    end_time = pd.Timestamp.now()
    time_span = end_time - start_time
    print("\n自定义超参数搜索完成, 结束时间:", end_time)
    print(f"总耗时: {time_span.total_seconds():.2f} 秒")
    print("最佳参数:", best_params)
    print(f"最佳RMSE (归一化后): {best_rmse:.4f}")
    print(f"最佳模型在 {best_model.early_stop_iter_} 迭代处早停。")

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
        best_model.predict(X_valid).reshape(-1, 1))
    y_valid_original = scaler_y.inverse_transform(y_valid.values.reshape(-1, 1))

    valid_mse = mean_squared_error(y_valid_original, y_pred_valid_original)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = mean_absolute_error(y_valid_original, y_pred_valid_original)
    valid_mape = (
        np.mean(np.abs((y_valid_original - y_pred_valid_original) / y_valid_original
        )) * 100
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
            "min_data_in_leaf": [best_params["min_data_in_leaf"]],
            "l2_leaf_reg": [best_params["l2_leaf_reg"]],
            "early_stop_iter": [best_model.early_stop_iter_],
            "总耗时（秒）": [time_span.total_seconds()],
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_GridSearch_{targets[0]}_Metrics_{current_time}.csv"
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
    plt.title("实际值 vs. 预测值 (原始尺度)")
    plt.grid(True)
    plt.show()
    df = pd.DataFrame(
        {
            f"实际 {targets[0]}": y_test_original.flatten(),
            f"预测 {targets[0]}": y_pred_original.flatten(),
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_GridSearch_{targets[0]}_Predictions_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存到: {output_file_path}")

    # 绘制最优模型的损失变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(best_model.train_rmse_history_)),
        best_model.train_rmse_history_,
        label="训练集 RMSE",
    )
    plt.plot(
        range(len(best_model.valid_rmse_history_)),
        best_model.valid_rmse_history_,
        label="验证集 RMSE",
    )
    plt.xlabel("迭代次数")
    plt.ylabel("RMSE")
    plt.title("最优模型的训练和验证集 RMSE 变化曲线")
    plt.legend()
    plt.grid(True)
    plt.show()

    df = pd.DataFrame(
        {
            "迭代次数": range(len(best_model.train_rmse_history_)),
            "训练集 RMSE": best_model.train_rmse_history_,
            "验证集 RMSE": best_model.valid_rmse_history_,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_GridSearch_{targets[0]}_RMSE_History_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\nRMSE 历史记录已保存到: {output_file_path}")

    df = pd.DataFrame(
        {
            "搜索轮次": range(n_iterations),
            "learning_rate": learning_rates,
            "depth": depths,
            "min_data_in_leaf": min_data_in_leafs,
            "l2_leaf_reg": l2_leaf_regs,
            "early_stop_iter": early_stop_iters,
            "valid_rmse": valid_rmses,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"CatBoost_GridSearch_{targets[0]}_Parameters_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n超参数记录已保存到: {output_file_path}")
