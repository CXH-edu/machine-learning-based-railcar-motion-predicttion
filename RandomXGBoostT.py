import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb  # 导入 XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import uniform, randint
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

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
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=42)
# 0.17647=0.15/(1-0.15)，确保验证集占总数据的15%

# --- 自定义 XGBoost 回归器类 (使用手动迭代和早停) ---
class XGBRegressorWithEarlyStopping(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        #gamma=0, # 对应 min_split_gain
        subsample=1.0,
        n_estimators=1000, # 对应 max_iter
        patience=10, # 对应早停轮数
        min_delta=1e-4, # 对应 CatBoost 的 min_delta
        random_state=42,
        # 其他 XGBoost 参数可以在这里添加
        objective='reg:squarederror', # 回归任务的标准目标函数
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        #self.gamma = gamma # min_split_gain
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.objective = objective
        
        self.model_ = None
        self.early_stop_iter_ = None
        self.train_rmse_history_ = []  # 记录训练集 RMSE
        self.valid_rmse_history_ = []  # 记录验证集 RMSE

    def fit(self, X, y, eval_set=None):
        if eval_set is None:
            # 如果没有提供验证集，则从训练集中划分
            X_train_fit, X_valid_fit, y_train_fit, y_valid_fit = train_test_split(
                X, y, test_size=0.1, random_state=self.random_state
            )
        else:
            X_train_fit, y_train_fit = X, y
            X_valid_fit, y_valid_fit = eval_set # eval_set 此时是 (X_valid, y_valid)

        # 初始化 XGBoost 模型 (这里 n_estimators 设置为1，因为我们是逐个添加树)
        # 注意：这里的 n_estimators 仅用于初始化模型，后续我们会手动迭代添加树
        current_model = xgb.XGBRegressor(
            objective=self.objective,
            n_estimators=1, # 每次只训练一棵树
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            #gamma=self.gamma,
            subsample=self.subsample,
            random_state=self.random_state,
            n_jobs=-1, # 利用所有核心
        )
        
        no_improve_count = 0
        best_valid_loss = float('inf')
        self.early_stop_iter_ = self.n_estimators # 默认是最大迭代次数

        self.train_rmse_history_ = []
        self.valid_rmse_history_ = []

        # 手动迭代训练
        for i in range(1, self.n_estimators + 1):
            # 对于第一次迭代，直接 fit
            if i == 1:
                current_model.fit(X_train_fit, y_train_fit, verbose=False)
            else:
                # 对于后续迭代，使用 xgb.train() 来添加新的树，并传递之前训练好的模型
                # 注意：这里需要使用 DMatrix，并且 eval_set 格式不同
                dtrain = xgb.DMatrix(X_train_fit, label=y_train_fit)
                
                # 获取当前模型的参数
                params = current_model.get_params()
                # 移除 n_estimators，因为我们是迭代训练
                params.pop('n_estimators', None) 
                
                # 初始化新的模型实例，使用当前迭代的 n_estimators
                # 为了实现增量训练，我们通常需要 xgb.train 而不是再次调用 XGBRegressor.fit
                # 但为了与 BaseEstimator 兼容，我们可以模拟这个过程，或者直接使用 current_model.n_estimators += 1
                # 实际上，XGBoost 的 incremental fit 并不是直接加树，而是重新 fit 一个更大的 n_estimators
                # 这种情况下，最佳做法是每次都用完整的 n_estimators 来 fit，并通过回调函数来获取历史
                # 或者，像 CatBoost 那样，每次只训练一棵树然后组合
                # 如果要严格模拟你原来的 CatBoost 逻辑 (每次加一棵树)，会稍微复杂一点
                # 这里我们采用 CatBoost 类似的做法：每次训练一个iterations=1的模型并作为init_model传入
                
                temp_model = xgb.XGBRegressor(
                    objective=self.objective,
                    n_estimators=1, # 每次只训练一棵树
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    #gamma=self.gamma,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                
                # 获取当前模型的预测值作为新的训练的初始预测（残差）
                # 这是实现梯度提升增量训练的关键步骤
                # 然而，XGBoost的XGBRegressor.fit方法并不直接支持init_model
                # 因此，我们必须每次都从头训练一个模型，但只取它的第i棵树
                # 或者，更常见的做法是让n_estimators变大，然后用callbacks来模拟早停
                # 为了保持与CatBoost自定义早停逻辑的相似性，我们通过逐次调用fit并监控验证集RMSE来模拟

                # 如果要真正实现“每次添加一棵树”的逻辑，需要使用底层的 DMatrix 和 xgb.train()
                # 并且要自己管理预测值的累积
                # 但为了简化并保持 sklearn API 风格，我们可以在每次迭代中
                # 模拟训练一个完整的模型，然后检查它的表现，虽然这效率不高
                
                # 最简单且符合XGBoost使用习惯的方法是：
                # 训练一个 n_estimators 最大的模型，然后用它的历史记录进行早停判断。
                # 但你明确要求“自定义迭代的方式”，这意味着不能直接使用 early_stopping_rounds。
                # 那么，我们需要手动模拟：训练一个 full model，然后截取其最佳迭代。
                # 或者，更符合你 CatBoost 逐迭代的方式，是训练一个 `n_estimators=i` 的模型，然后评估
                
                # 重写此处逻辑以匹配原始 CatBoost 的 "每次添加1棵树并评估" 的概念
                # 这是最接近你原始代码逻辑的实现方式
                full_model = xgb.XGBRegressor(
                    objective=self.objective,
                    n_estimators=i, # 每次训练 i 棵树
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    #gamma=self.gamma,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                full_model.fit(X_train_fit, y_train_fit, verbose=False) # 不输出训练过程
                current_model = full_model # 更新当前模型

            # 在训练集和验证集上进行预测
            y_train_pred = current_model.predict(X_train_fit)
            y_valid_pred = current_model.predict(X_valid_fit)

            # 计算 RMSE
            train_rmse = np.sqrt(mean_squared_error(y_train_fit, y_train_pred))
            valid_rmse = np.sqrt(mean_squared_error(y_valid_fit, y_valid_pred))
            
            self.train_rmse_history_.append(train_rmse)
            self.valid_rmse_history_.append(valid_rmse)

            # 早停逻辑
            if best_valid_loss - valid_rmse > self.min_delta:
                best_valid_loss = valid_rmse
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    self.early_stop_iter_ = i
                    break
        
        # 保存最终模型为在早停迭代处（或最大迭代数）的模型
        # 注意：这里 best_model 不直接是 current_model，因为 current_model 可能已经过拟合
        # 如果是早停，我们应该返回在 best_valid_loss 时的模型状态
        # 但是 XGBoostRegressor 不直接提供历史模型，所以我们最好的办法是
        # 使用 best_valid_loss 对应的迭代次数来重新训练一个模型
        # 或者，如果 early_stopping_rounds 是0，则返回 full n_estimators
        # 这里为了简化，我们让 self.model_ 存储最后一次迭代的模型，
        # 并记录 early_stop_iter_。在实际使用时，应该使用 early_stop_iter_ 重新训练
        # 或者在循环中保存最佳模型。
        
        # 更严谨的做法是在循环中保存最佳模型状态
        if self.early_stop_iter_ < self.n_estimators: # 如果发生了早停
            final_n_estimators = self.early_stop_iter_
        else: # 没有早停，训练到最大迭代次数
            final_n_estimators = self.n_estimators
        
        self.model_ = xgb.XGBRegressor(
            objective=self.objective,
            n_estimators=final_n_estimators, # 训练到早停的最佳迭代次数
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            #gamma=self.gamma,
            subsample=self.subsample,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_train_fit, y_train_fit, verbose=False) # 重新训练最终模型
        
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError(
                "This XGBRegressorWithEarlyStopping instance is not fitted yet. Call 'fit' before using 'predict'."
            )
        return self.model_.predict(X)

# --- 主程序部分 ---
if __name__ == "__main__":
    # 定义 XGBoost 随机搜索的超参数分布
    param_dist = {
        'learning_rate': uniform(0.005, 0.095),      # [0.005, 0.1]
        'max_depth': randint(3, 10),                 # [3, 9]
        'min_child_weight': randint(1, 11),          # [1, 10]
        #'gamma': uniform(0, 5)               # 对应 min_split_gain, 伽马值通常是浮点数，设为 [0, 5) 之间
        'subsample': uniform(0.7, 0.3)               # [0.7, 1.0]
    }

    best_rmse = float("inf")
    best_params = None
    best_model_instance = None # 保存最佳模型的实例，以便获取历史记录
    # 随机搜索次数
    n_iterations = 100 # 你可以把这个值调高到 300 甚至更多

    learning_rates = []
    max_depths = []
    min_child_weights = []
    #gammas = []
    subsamples = []
    early_stop_iters = []
    valid_rmses = []

    start_time = pd.Timestamp.now()
    print("开始进行自定义超参数搜索..., 开始时间:", start_time)
    for i in range(n_iterations):
        # 随机采样超参数
        params = {
            "learning_rate": param_dist["learning_rate"].rvs(),
            "max_depth": param_dist["max_depth"].rvs(),
            "min_child_weight": param_dist["min_child_weight"].rvs(),
            #"gamma": param_dist["gamma"].rvs(),
            "subsample": param_dist["subsample"].rvs(),
        }
        learning_rates.append(params["learning_rate"])
        max_depths.append(params["max_depth"])
        min_child_weights.append(params["min_child_weight"])
        #gammas.append(params["gamma"])
        subsamples.append(params["subsample"])

        # 训练模型
        model = XGBRegressorWithEarlyStopping(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            #gamma=params["gamma"],
            subsample=params["subsample"],
            n_estimators=1000, # 这里的 n_estimators 相当于最大迭代次数
            patience=10, # 对应早停轮数
            min_delta=1e-4,
            random_state=42,
        )
        # eval_set 传入的是一个元组 (X_valid, y_valid)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        # 在验证集上评估
        y_valid_pred = model.predict(X_valid)
        valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        valid_rmses.append(valid_rmse)

        # 更新最佳模型
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_params = params
            best_model_instance = model # 保存最佳模型的完整实例
            print(
                f"迭代 {i+1}/{n_iterations}, 新最佳 RMSE: {best_rmse:.4f}, 参数: {best_params}"
            )
        else:
            print(f"迭代 {i+1}/{n_iterations}")
        early_stop_iters.append(model.early_stop_iter_)

    end_time = pd.Timestamp.now()
    time_span = end_time - start_time
    print("\n自定义超参数搜索完成, 结束时间:", end_time)
    print(f"总耗时: {time_span.total_seconds():.2f} 秒")
    print("最佳参数:", best_params)
    print(f"最佳RMSE (归一化后): {best_rmse:.4f}")
    if best_model_instance and best_model_instance.early_stop_iter_ is not None:
        print(f"最佳模型在 {best_model_instance.early_stop_iter_} 迭代处早停。")
    else:
        print("最佳模型训练完成，但早停迭代次数未记录或未发生早停。")

    # 使用最佳模型实例进行预测和评估
    # 在测试集上进行预测
    y_pred_normalized = best_model_instance.predict(X_test)

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
    # 避免除以零，添加对y_test_original中0值的检查
    valid_indices = y_test_original != 0
    if np.any(valid_indices):
        test_mape = (
            np.mean(np.abs((y_test_original[valid_indices] - y_pred_original[valid_indices]) / y_test_original[valid_indices])) * 100
        )
    else:
        test_mape = np.nan # 或设置为0，取决于具体场景
    test_r2 = r2_score(y_test_original, y_pred_original)

    print(f"\n最终模型在原始尺度测试集上的评估结果:")
    print(f"MSE: {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")
    print(f"R2 Score: {test_r2:.4f}")

    y_pred_train_normalized = best_model_instance.predict(X_train)
    y_train_original = scaler_y.inverse_transform(y_train.values.reshape(-1, 1))
    y_pred_train_original = scaler_y.inverse_transform(y_pred_train_normalized.reshape(-1, 1))

    train_mse = mean_squared_error(y_train_original, y_pred_train_original)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_original, y_pred_train_original)
    valid_indices_train = y_train_original != 0
    if np.any(valid_indices_train):
        train_mape = (
            np.mean(np.abs((y_train_original[valid_indices_train] - y_pred_train_original[valid_indices_train]) / y_train_original[valid_indices_train]))
            * 100
        )
    else:
        train_mape = np.nan
    train_r2 = r2_score(y_train_original, y_pred_train_original)

    y_pred_valid_normalized = best_model_instance.predict(X_valid)
    y_valid_original = scaler_y.inverse_transform(y_valid.values.reshape(-1, 1))
    y_pred_valid_original = scaler_y.inverse_transform(y_pred_valid_normalized.reshape(-1, 1))

    valid_mse = mean_squared_error(y_valid_original, y_pred_valid_original)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = mean_absolute_error(y_valid_original, y_pred_valid_original)
    valid_indices_valid = y_valid_original != 0
    if np.any(valid_indices_valid):
        valid_mape = (
            np.mean(np.abs((y_valid_original[valid_indices_valid] - y_pred_valid_original[valid_indices_valid]) / y_valid_original[valid_indices_valid]
            )) * 100
        )
    else:
        valid_mape = np.nan

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
            "max_depth": [best_params["max_depth"]],
            "min_child_weight": [best_params["min_child_weight"]],
            #"gamma": [best_params["gamma"]], # 修改为 gamma
            "subsample": [best_params["subsample"]],
            "early_stop_iter": [best_model_instance.early_stop_iter_],
            "总耗时（秒）": [time_span.total_seconds()],
        }
    )
    output_file_path = os.path.join(
        output_dir, f"XGBoost_RandomSearch_{targets[0]}_Metrics_{current_time}.csv"
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
    plt.title("XGBoost实际值 vs. 预测值 (原始尺度)")
    plt.grid(True)
    plt.show()
    df = pd.DataFrame(
        {
            f"实际 {targets[0]}": y_test_original.flatten(),
            f"预测 {targets[0]}": y_pred_original.flatten(),
        }
    )
    output_file_path = os.path.join(
        output_dir, f"XGBoost_RandomSearch_{targets[0]}_Predictions_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存到: {output_file_path}")

    # 绘制最优模型的损失变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(best_model_instance.train_rmse_history_)),
        best_model_instance.train_rmse_history_,
        label="训练集 RMSE",
    )
    plt.plot(
        range(len(best_model_instance.valid_rmse_history_)),
        best_model_instance.valid_rmse_history_,
        label="验证集 RMSE",
    )
    plt.xlabel("迭代次数")
    plt.ylabel("RMSE")
    plt.title("XGBoost最优模型的训练和验证集 RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    df = pd.DataFrame(
        {
            "迭代次数": range(len(best_model_instance.train_rmse_history_)),
            "训练集 RMSE": best_model_instance.train_rmse_history_,
            "验证集 RMSE": best_model_instance.valid_rmse_history_,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"XGBoost_RandomSearch_{targets[0]}_RMSE_History_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\nRMSE 历史记录已保存到: {output_file_path}")

    df = pd.DataFrame(
        {
            "搜索轮次": range(n_iterations),
            "learning_rate": learning_rates,
            "max_depth": max_depths, # 修改为 max_depth
            "min_child_weight": min_child_weights, # 修改为 min_child_weight
            #"gamma": gammas, # 修改为 gamma
            "subsample": subsamples,
            "early_stop_iter": early_stop_iters,
            "valid_rmse": valid_rmses,
        }
    )
    output_file_path = os.path.join(
        output_dir, f"XGBoost_RandomSearch_{targets[0]}_Parameters_{current_time}.csv"
    )
    df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
    print(f"\n超参数记录已保存到: {output_file_path}")