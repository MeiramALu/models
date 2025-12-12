import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Устанавливаем бэкенд AGG для корректной работы Matplotlib в веб-сервере
plt.switch_backend('AGG')

# ==========================================
# 1. Вспомогательные функции для Графики
# ==========================================

def get_graph():
    """Конвертирует текущий график Matplotlib в строку Base64 для HTML"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close() # Обязательно закрываем фигуру, чтобы не забивать память
    return graph

# ==========================================
# 2. Функции обработки данных
# ==========================================

def check_nulls(df):
    """Возвращает DataFrame со статистикой пропусков"""
    df_null = pd.DataFrame(
        {
            "Missing Values": df.isna().sum().values,
            "% of Total Values": 100 * df.isnull().sum() / len(df),
        },
        index=df.isna().sum().index,
    )
    df_null = df_null.sort_values("% of Total Values", ascending=False).round(2)
    return df_null

def delete_columns(data, columns):
    """Удаляет указанные колонки"""
    # Проверка, является ли columns списком или одной строкой
    if isinstance(columns, str):
        columns = [columns]
    dataframe = data.drop(columns, axis=1)
    return dataframe

def fill_numerical_data(data, columns):
    """Заполняет пропуски в числовых колонках медианой"""
    for col in columns:
        data[col] = data[col].fillna(data[col].median())
    return data

def fill_categorical_data(data, columns):
    """Заполняет пропуски в категориальных колонках модой"""
    imputer = SimpleImputer(strategy="most_frequent")
    for col in columns:
        # ravel() нужен для приведения формата
        data[col] = imputer.fit_transform(data[[col]]).ravel()
    return data

def apply_one_hot_encoder(df, columns):
    dataframe = pd.get_dummies(df, columns=columns, drop_first=True)
    return dataframe

def apply_label_encoder(df, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

def apply_ordinal_encoder(df, columns):
    ordinal_encoder = OrdinalEncoder()
    for column in columns:
        df[column] = ordinal_encoder.fit_transform(df[[column]])
    return df

def apply_min_max_scaling(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def apply_standardization(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def apply_log_transformation(df, columns):
    for column in columns:
        df[column] = np.log(df[column] + 1)
    return df

def custom_train_test_split(X, y, size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    return X_train, X_test, y_train, y_test

# ==========================================
# 3. Функции Визуализации (Возвращают Base64 string)
# ==========================================

# Примечание: Altair заменен на Seaborn/Matplotlib для унификации вывода картинок в Django

def line_chart(data, x_axis, y_axis, hue=None):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=x_axis, y=y_axis, hue=hue)
    plt.title(f"Line Plot: {x_axis} vs {y_axis}")
    return get_graph()

def scatter_plot(data, x_axis, y_axis, hue=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=hue)
    plt.title(f"Scatter Plot: {x_axis} vs {y_axis}")
    return get_graph()

def histogram(data, x_axis, hue=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=x_axis, hue=hue, kde=True)
    plt.title(f"Histogram: {x_axis}")
    return get_graph()

def box_plot(data, x_axis, y_axis, hue=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_axis, y=y_axis, hue=hue)
    plt.title(f"Box Plot: {x_axis} vs {y_axis}")
    return get_graph()

def density_plot(data, x_axis, hue=None):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=x_axis, hue=hue, fill=True)
    plt.title(f"Density Plot: {x_axis}")
    return get_graph()

def correlation_matrix_visualize(data, color="viridis"):
    plt.figure(figsize=(12, 10))
    # Выбираем только числовые колонки для корреляции
    numeric_df = data.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap=color, fmt=".2f")
    plt.title("Correlation of Features")
    return get_graph()

def confusion_matrix_visualization(y_true, y_pred, input_color="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=input_color, cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    return get_graph()

def regression_result_visualization(y_true, y_pred, input_color="#00f900"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, c=input_color)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Regression Results: True vs Predicted")
    # Добавим линию идеального предсказания
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    return get_graph()

# ==========================================
# 4. Модели Машинного Обучения (Factory functions)
# ==========================================

def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred

def custom_linear_regression():
    model = LinearRegression()
    return model

def custom_logistic_regression(C_c=1.0, n_jobs_c=-1, max_iter_c=100, l1_ratio_c=None):
    # l1_ratio используется только если penalty='elasticnet', по умолчанию l2.
    # Чтобы избежать ошибок, добавим проверку или оставим дефолтные параметры, если не elasticnet
    model = LogisticRegression(
        C=C_c, n_jobs=n_jobs_c, max_iter=max_iter_c, l1_ratio=l1_ratio_c
    )
    return model

def custom_decision_tree_regression(
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeRegressor(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_decision_tree_classification(
    criterion_c="gini",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = DecisionTreeClassifier(
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_random_forest_regression(
    n_estimators_c=100,
    criterion_c="squared_error",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = RandomForestRegressor(
        n_estimators=n_estimators_c,
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_random_forest_classification(
    n_estimators_c=100,
    criterion_c="gini",
    max_depth_c=None,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators_c,
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_svr(kernel_c="rbf", degree_c=3):
    model = SVR(kernel=kernel_c, degree=degree_c)
    return model

def custom_svc(kernel_c="rbf", degree_c=3):
    model = SVC(kernel=kernel_c, degree=degree_c)
    return model

def custom_gbc(
    loss_c="log_loss",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="friedman_mse",
    max_depth_c=3,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingClassifier(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_gbr(
    loss_c="squared_error",
    learning_rate_c=0.1,
    n_estimators_c=100,
    criterion_c="friedman_mse",
    max_depth_c=3,
    min_samples_split_c=2,
    min_samples_leaf_c=1,
    ccp_alpha_c=0.0,
):
    model = GradientBoostingRegressor(
        loss=loss_c,
        learning_rate=learning_rate_c,
        n_estimators=n_estimators_c,
        criterion=criterion_c,
        max_depth=max_depth_c,
        min_samples_split=min_samples_split_c,
        min_samples_leaf=min_samples_leaf_c,
        ccp_alpha=ccp_alpha_c,
    )
    return model

def custom_mlp_classifier(
    activation_func_c="relu",
    solver_c="adam",
    number_layers_c=100
):
    # hidden_layer_sizes ожидает кортеж, если один слой - (N,)
    model = MLPClassifier(
        activation=activation_func_c,
        solver=solver_c,
        hidden_layer_sizes=(number_layers_c,),
    )
    return model

def custom_mlp_regressor(
    activation_func_c="relu",
    solver_c="adam",
):
    model = MLPRegressor(
        activation=activation_func_c,
        solver=solver_c
    )
    return model