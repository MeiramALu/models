import os
import time  # Добавлено для генерации уникальных имен файлов истории
import pandas as pd
import joblib
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, FileResponse
from django.contrib import messages

# Импорт наших модулей
from .models import UploadedDataset
from .forms import (
    UploadFileForm,
    DeleteColumnForm,
    FillMissingForm,
    EncodingForm,
    ScalingForm,
    VisualizeForm,
    TrainModelForm
)
from .utils import (
    check_nulls, delete_columns, fill_numerical_data, fill_categorical_data,
    apply_one_hot_encoder, apply_label_encoder, apply_ordinal_encoder,
    apply_min_max_scaling, apply_standardization, apply_log_transformation,
    line_chart, scatter_plot, histogram, box_plot, density_plot, correlation_matrix_visualize,
    custom_train_test_split, train_and_predict, regression_result_visualization, confusion_matrix_visualization,
    # Factory functions для моделей
    custom_linear_regression, custom_logistic_regression,
    custom_decision_tree_regression, custom_decision_tree_classification,
    custom_random_forest_regression, custom_random_forest_classification,
    custom_svr, custom_svc, custom_gbc, custom_gbr,
    custom_mlp_classifier, custom_mlp_regressor
)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, \
    precision_score, recall_score


# ==========================================
# Вспомогательные функции (State Management)
# ==========================================

def get_df(request):
    """
    Получает текущий DataFrame для сессии пользователя.
    Возвращает кортеж: (DataFrame, DatasetObject) или (None, None)
    """
    if not request.session.session_key:
        request.session.save()

    session_key = request.session.session_key
    try:
        dataset = UploadedDataset.objects.get(session_key=session_key)
        # Получаем путь к актуальному файлу (оригинал или обработанный)
        file_path = dataset.get_active_file_path()

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df, dataset
        else:
            # Если файл физически удален, но запись в БД есть
            dataset.delete()
            return None, None
    except UploadedDataset.DoesNotExist:
        return None, None


def save_df(request, df):
    """
    Сохраняет измененный DataFrame как 'processed_file'
    """
    session_key = request.session.session_key
    dataset = UploadedDataset.objects.get(session_key=session_key)

    # Формируем путь сохранения
    filename = f"processed_{session_key}.csv"
    # Папка media/datasets/processed/
    save_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'processed')
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)

    # Сохраняем CSV
    df.to_csv(full_path, index=False)

    # Обновляем поле в БД (путь относительно MEDIA_ROOT)
    dataset.processed_file.name = os.path.join('datasets', 'processed', filename)
    dataset.save()


# ==========================================
# НОВАЯ ЛОГИКА: История изменений (Undo)
# ==========================================

def save_to_history(request, df):
    """
    Сохраняет текущее состояние DataFrame в историю ПЕРЕД изменением.
    """
    session_key = request.session.session_key

    # 1. Создаем папку для истории
    history_dir = os.path.join(settings.MEDIA_ROOT, 'datasets', 'history')
    os.makedirs(history_dir, exist_ok=True)

    # 2. Генерируем уникальное имя (timestamp)
    filename = f"hist_{session_key}_{int(time.time() * 1000)}.csv"
    full_path = os.path.join(history_dir, filename)

    # 3. Сохраняем файл
    df.to_csv(full_path, index=False)

    # 4. Добавляем путь в сессию (стек)
    if 'history_stack' not in request.session:
        request.session['history_stack'] = []

    # (Опционально) Ограничим историю 10 шагами, чтобы не забивать память
    if len(request.session['history_stack']) >= 10:
        oldest_file = request.session['history_stack'].pop(0)
        if os.path.exists(oldest_file):
            try:
                os.remove(oldest_file)
            except OSError:
                pass

    request.session['history_stack'].append(full_path)
    request.session.modified = True


def perform_undo(request):
    """
    Возвращает состояние на шаг назад.
    """
    if 'history_stack' in request.session and request.session['history_stack']:
        # 1. Достаем последний путь
        last_file_path = request.session['history_stack'].pop()
        request.session.modified = True

        if os.path.exists(last_file_path):
            # 2. Читаем старый файл
            df = pd.read_csv(last_file_path)

            # 3. Сохраняем его как ТЕКУЩИЙ (processed)
            save_df(request, df)

            # 4. Удаляем файл истории (так как мы его "использовали")
            try:
                os.remove(last_file_path)
            except OSError:
                pass
            return True
    return False


def clear_history(request):
    """Очищает всю историю (при сбросе или новой загрузке)"""
    if 'history_stack' in request.session:
        for path in request.session['history_stack']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        request.session['history_stack'] = []
        request.session.modified = True


# ==========================================
# Views (Страницы)
# ==========================================

def home(request):
    return render(request, 'core/home.html')


def upload_data(request):
    if not request.session.session_key:
        request.session.save()

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Удаляем старый датасет этой сессии, если был
            UploadedDataset.objects.filter(session_key=request.session.session_key).delete()

            # Создаем новый
            instance = UploadedDataset(
                session_key=request.session.session_key,
                file=request.FILES['file']
            )
            instance.save()

            # ВАЖНО: Очищаем историю при загрузке нового файла
            clear_history(request)

            messages.success(request, "Файл успешно загружен!")
            return redirect('edit_data')
    else:
        form = UploadFileForm()

    return render(request, 'core/upload.html', {'form': form})


def edit_data(request):
    df, dataset = get_df(request)
    if df is None:
        return redirect('upload_data')

    columns = df.columns.tolist()

    # Инициализация форм
    delete_form = DeleteColumnForm(request.POST or None, columns=columns, prefix='del')
    fill_form = FillMissingForm(request.POST or None, columns=columns, prefix='fill')
    encoding_form = EncodingForm(request.POST or None, columns=columns, prefix='enc')
    scaling_form = ScalingForm(request.POST or None, prefix='scale')

    # Проверяем наличие истории для кнопки "Назад"
    has_history = len(request.session.get('history_stack', [])) > 0

    # Обработка действий (POST)
    if request.method == 'POST':

        # === ЛОГИКА UNDO (ШАГ НАЗАД) ===
        if 'btn_undo' in request.POST:
            if perform_undo(request):
                messages.info(request, "Возврат к предыдущему состоянию выполнен.")
            else:
                messages.warning(request, "Нет действий для отмены.")
            return redirect('edit_data')

        # === СБРОС ИЗМЕНЕНИЙ ===
        elif 'btn_reset' in request.POST:
            session_key = request.session.session_key
            try:
                dataset = UploadedDataset.objects.get(session_key=session_key)
                dataset.processed_file = None
                dataset.save()
                # Очищаем историю
                clear_history(request)
                messages.warning(request, "Все изменения сброшены к оригиналу.")
            except UploadedDataset.DoesNotExist:
                pass
            return redirect('edit_data')

        # === ОПЕРАЦИИ РЕДАКТИРОВАНИЯ ===

        # 1. Удаление колонок
        if 'btn_delete' in request.POST and delete_form.is_valid():
            cols = delete_form.cleaned_data['columns_to_delete']
            if cols:
                save_to_history(request, df)  # <--- СОХРАНЯЕМ ПЕРЕД ИЗМЕНЕНИЕМ
                df = delete_columns(df, cols)
                save_df(request, df)
                messages.success(request, f"Удалены колонки: {', '.join(cols)}")
                return redirect('edit_data')

        # 2. Заполнение пропусков
        elif 'btn_fill' in request.POST and fill_form.is_valid():
            cols = fill_form.cleaned_data['columns_to_fill']
            if cols:
                save_to_history(request, df)  # <--- СОХРАНЯЕМ ПЕРЕД ИЗМЕНЕНИЕМ

                # Определяем тип колонок для правильного заполнения
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
                cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

                if num_cols:
                    df = fill_numerical_data(df, num_cols)
                if cat_cols:
                    df = fill_categorical_data(df, cat_cols)

                save_df(request, df)
                messages.success(request, "Пропуски заполнены.")
                return redirect('edit_data')

        # 3. Кодирование (Encoding)
        elif 'btn_encode' in request.POST and encoding_form.is_valid():
            save_to_history(request, df)  # <--- СОХРАНЯЕМ ПЕРЕД ИЗМЕНЕНИЕМ

            method = encoding_form.cleaned_data['encoding_type']
            cols = encoding_form.cleaned_data['columns_to_encode']

            if method == 'one_hot':
                df = apply_one_hot_encoder(df, cols)
            elif method == 'label':
                df = apply_label_encoder(df, cols)
            elif method == 'ordinal':
                df = apply_ordinal_encoder(df, cols)

            save_df(request, df)
            messages.success(request, f"Применено кодирование: {method}")
            return redirect('edit_data')

        # 4. Масштабирование (Scaling)
        elif 'btn_scale' in request.POST and scaling_form.is_valid():
            save_to_history(request, df)  # <--- СОХРАНЯЕМ ПЕРЕД ИЗМЕНЕНИЕМ

            method = scaling_form.cleaned_data['scaling_method']
            # Применяем только к числовым колонкам
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if method == 'normalization':
                df = apply_min_max_scaling(df, numeric_cols)
            elif method == 'standardization':
                df = apply_standardization(df, numeric_cols)
            elif method == 'log':
                df = apply_log_transformation(df, numeric_cols)

            save_df(request, df)
            messages.success(request, f"Применено масштабирование: {method}")
            return redirect('edit_data')

    # Контекст для шаблона
    null_stats = check_nulls(df)

    context = {
        'df_head': df.head().to_html(classes="table table-striped table-sm", index=False),
        'df_shape': df.shape,
        'df_describe': df.describe().to_html(classes="table table-bordered table-sm"),
        'null_stats': null_stats.to_html(classes="table table-sm") if not null_stats.empty else "Нет пропусков",
        'delete_form': delete_form,
        'fill_form': fill_form,
        'encoding_form': encoding_form,
        'scaling_form': scaling_form,
        'has_history': has_history,  # Флаг для кнопки "Назад"
    }
    return render(request, 'core/edit.html', context)


def visualize_data(request):
    df, _ = get_df(request)
    if df is None: return redirect('upload_data')

    chart_image = None
    # Корреляционная матрица (генерируем сразу)
    corr_matrix_image = correlation_matrix_visualize(df)

    form = VisualizeForm(request.POST or None, columns=df.columns.tolist())

    if request.method == 'POST' and form.is_valid():
        plot_type = form.cleaned_data['plot_type']
        x_axis = form.cleaned_data['x_axis']
        y_axis = form.cleaned_data['y_axis']
        hue = form.cleaned_data['hue']
        if hue == '': hue = None
        # Цветовая схема используется внутри utils (в коде utils выше я захардкодил 'viridis', 
        # но можно передать аргумент, если обновить функции в utils)

        if plot_type == 'Line Plot':
            chart_image = line_chart(df, x_axis, y_axis, hue)
        elif plot_type == 'Scatter Plot':
            chart_image = scatter_plot(df, x_axis, y_axis, hue)
        elif plot_type == 'Histogram Plot':
            chart_image = histogram(df, x_axis, hue)
        elif plot_type == 'Box Plot':
            chart_image = box_plot(df, x_axis, y_axis, hue)
        elif plot_type == 'Density Plot':
            chart_image = density_plot(df, x_axis, hue)

    return render(request, 'core/visualize.html', {
        'form': form,
        'chart_image': chart_image,
        'corr_matrix_image': corr_matrix_image
    })


def train_model(request):
    df, _ = get_df(request)
    if df is None: return redirect('upload_data')

    results = {}
    result_plot = None
    form = TrainModelForm(request.POST or None, columns=df.columns.tolist())

    if request.method == 'POST' and form.is_valid():
        data = form.cleaned_data
        target = data['target_column']
        task_type = data['task_type']
        algo = data['algorithm']
        test_size = data['test_size']

        # Подготовка данных
        try:
            X = delete_columns(df, target)
            y = df[target]
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, size=test_size)
        except Exception as e:
            messages.error(request, f"Ошибка при разделении данных: {e}")
            return redirect('train_model')

        # Выбор и инициализация модели
        model = None

        # Получаем гиперпараметры
        n_est = data.get('n_estimators', 100)
        depth = data.get('max_depth', None)
        c_val = data.get('C_param', 1.0)
        kern = data.get('kernel', 'rbf')
        lr = data.get('learning_rate', 0.1)

        # Фабрика моделей (Logic mapping)
        if algo == 'Linear Regression':
            model = custom_linear_regression()
        elif algo == 'Logistic Regression':
            model = custom_logistic_regression(C_c=c_val)

        elif algo == 'Decision Tree Regression':
            model = custom_decision_tree_regression(max_depth_c=depth)
        elif algo == 'Decision Tree CLassifier':
            model = custom_decision_tree_classification(max_depth_c=depth)

        elif algo == 'Random Forest Regression':
            model = custom_random_forest_regression(n_estimators_c=n_est, max_depth_c=depth)
        elif algo == 'Random Forest Classification':
            model = custom_random_forest_classification(n_estimators_c=n_est, max_depth_c=depth)

        elif algo == 'Support Vector Machine Regression':
            model = custom_svr(kernel_c=kern)
        elif algo == 'Support Vector Machine CLassification':
            model = custom_svc(kernel_c=kern)

        elif algo == 'Gradient Boosting Regression':
            model = custom_gbr(n_estimators_c=n_est, learning_rate_c=lr, max_depth_c=depth)
        elif algo == 'Gradient Boosting Classification':
            model = custom_gbc(n_estimators_c=n_est, learning_rate_c=lr, max_depth_c=depth)

        elif algo == 'Multi Layer Perceptron Regression':
            model = custom_mlp_regressor()
        elif algo == 'Multi Layer Perceptron Classifier':
            model = custom_mlp_classifier()

        # Обучение
        try:
            model, pred = train_and_predict(model, X_train, y_train, X_test)

            # Метрики и Графики
            if task_type == 'Regression':
                results['R2'] = round(r2_score(y_test, pred), 4)
                results['MAE'] = round(mean_absolute_error(y_test, pred), 4)
                results['MSE'] = round(mean_squared_error(y_test, pred), 4)
                result_plot = regression_result_visualization(y_test, pred)
            else:
                results['Accuracy'] = round(accuracy_score(y_test, pred), 4)
                # Обработка multiclass для F1/Precision/Recall
                avg_method = 'binary' if y.nunique() == 2 else 'weighted'
                results['F1'] = round(f1_score(y_test, pred, average=avg_method), 4)
                results['Precision'] = round(precision_score(y_test, pred, average=avg_method), 4)
                results['Recall'] = round(recall_score(y_test, pred, average=avg_method), 4)
                result_plot = confusion_matrix_visualization(y_test, pred)

            # Сохранение модели
            model_filename = f"model_{request.session.session_key}.joblib"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', model_filename)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)

            # Сохраняем путь в сессии для скачивания
            request.session['model_file'] = model_path
            messages.success(request, "Модель успешно обучена!")

        except Exception as e:
            messages.error(request, f"Ошибка при обучении: {e}")

    return render(request, 'core/train.html', {
        'form': form,
        'results': results,
        'result_plot': result_plot,
        'model_ready': 'model_file' in request.session
    })


def download_model(request):
    """Скачивание обученной модели"""
    model_path = request.session.get('model_file')
    if model_path and os.path.exists(model_path):
        response = FileResponse(open(model_path, 'rb'), as_attachment=True, filename="trained_model.joblib")
        return response
    else:
        messages.error(request, "Файл модели не найден. Сначала обучите модель.")
        return redirect('train_model')