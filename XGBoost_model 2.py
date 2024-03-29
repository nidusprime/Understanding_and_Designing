import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from skopt import BayesSearchCV

# 读取数据
data = pd.read_csv("data/data2.csv")

# 定义目标列
target_columns = ['water permeability (LMH/bar)', 'organic compound removal (%)',
                  'flux recovery ratio(%)', 'reversible fouling ratio (%)', 'irreversible fouling ratio (%)']

# 对分类特征进行独热编码
categorical_features = ['organic compound', 'foulant']
data_encoded = pd.get_dummies(data, columns=categorical_features)

# 定义特征列
feature_columns = [col for col in data_encoded.columns if col not in target_columns]

# 获取特征和目标变量
x = data_encoded[feature_columns].values  # 将特征转换为数组格式
y = data[target_columns]

# 分割数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 创建一个字典，用于存储每个目标的模型、评价指标和超参数
models = {}
evaluation_metrics = {}
hyperparameters = {}

# 遍历每个目标进行模型训练和评估
for target in target_columns:
    print(f"Training model for {target}")

    # 构建XGBoost回归模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 定义超参数搜索空间
    param_space = {
        'colsample_bytree': (0, 1),
        'learning_rate': (0, 1),
        'max_depth': (1, 6),
        'subsample': (0, 1),
        'reg_alpha': (0, 10),
        'gamma': (0, 20),
        'reg_lambda': (1, 10),
        'n_estimators': (1, 100),
    }

    # 使用贝叶斯优化搜索最佳超参数
    opt = BayesSearchCV(model, param_space, cv=5, n_iter=50, scoring='neg_mean_squared_error', random_state=42,
                        verbose=1)

    # 训练模型
    opt.fit(x_train, y_train[target])

    # 在训练集上做预测
    y_train_pred = opt.predict(x_train)

    # 计算训练集的R2和RMSE
    train_r2 = r2_score(y_train[target], y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train[target], y_train_pred))

    # 在测试集上做预测
    y_test_pred = opt.predict(x_test)

    # 计算测试集的R2和RMSE
    test_r2 = r2_score(y_test[target], y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test[target], y_test_pred))

    # 将模型和评价指标存储在字典中
    models[target] = opt
    evaluation_metrics[target] = {'Train_R2': train_r2, 'Train_RMSE': train_rmse, 'Test_R2': test_r2,
                                  'Test_RMSE': test_rmse}

    # 存储最佳超参数
    hyperparameters[target] = opt.best_params_

    # 打印每个目标的评价指标和超参数
    print(
        f"Metrics for {target}: Train R2 = {train_r2}, Train RMSE = {train_rmse}, Test R2 = {test_r2}, Test RMSE = {test_rmse}")
    print(f"Best hyperparameters for {target}: {opt.best_params_}")

# 统一输出所有目标的评价指标和超参数
print("Evaluation Metrics and Hyperparameters:")
for target, metrics in evaluation_metrics.items():
    print(f"Target: {target}")
    print("Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("Hyperparameters:")
    print(hyperparameters[target])
    print()
