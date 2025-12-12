from django import forms
from django.core.validators import FileExtensionValidator

# ==========================================
# 1. Форма загрузки
# ==========================================

class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="Загрузите файл",
        help_text="Принимает только файлы формата .csv",
        validators=[FileExtensionValidator(allowed_extensions=['csv'])],
        widget=forms.ClearableFileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )


# ==========================================
# 2. Формы редактирования (Edit Page)
# ==========================================

class BaseDynamicForm(forms.Form):
    """
    Базовый класс для форм, которым нужно знать колонки датафрейма.
    Принимает список columns при инициализации.
    """

    def __init__(self, *args, columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        if columns:
            # Создаем список кортежей (название, название) для ChoiceField
            choices = [(col, col) for col in columns]

            # Проходим по всем полям и если они требуют выбора колонок, обновляем choices
            for field_name, field in self.fields.items():
                if hasattr(field, 'choices') and not field.choices:
                    # Если choices уже заданы жестко (статически), мы их не трогаем
                    field.choices = choices


class DeleteColumnForm(BaseDynamicForm):
    columns_to_delete = forms.MultipleChoiceField(
        label="Выберите параметры для удаления",
        # Используем чекбоксы (важно для нового дизайна)
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        required=False
    )


class FillMissingForm(BaseDynamicForm):
    """Форма для заполнения пропусков (общая для чисел и категорий)"""
    columns_to_fill = forms.MultipleChoiceField(
        label="Выберите колонки для заполнения (Медиана/Мода)",
        # Используем чекбоксы
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        required=False
    )


class EncodingForm(BaseDynamicForm):
    """Форма для кодирования категориальных данных"""
    ENCODING_TYPES = [
        ('one_hot', 'One Hot Encoding'),
        ('label', 'Label Encoding'),
        ('ordinal', 'Ordinal Encoding'),
    ]
    encoding_type = forms.ChoiceField(
        label="Метод кодирования",
        choices=ENCODING_TYPES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    columns_to_encode = forms.MultipleChoiceField(
        label="Выберите колонки",
        # Используем чекбоксы
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        required=True
    )


class ScalingForm(forms.Form):
    """Форма для масштабирования (стандартизация, нормализация)"""
    SCALING_TYPES = [
        ('normalization', 'Нормализация (MinMax)'),
        ('standardization', 'Стандартизация (StandardScaler)'),
        ('log', 'Логарифмическая трансформация'),
    ]
    scaling_method = forms.ChoiceField(
        label="Метод масштабирования",
        choices=SCALING_TYPES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )


# ==========================================
# 3. Форма визуализации
# ==========================================

class VisualizeForm(BaseDynamicForm):
    PLOT_TYPES = [
        ('Line Plot', 'Line Plot'),
        ('Scatter Plot', 'Scatter Plot'),
        ('Histogram Plot', 'Histogram Plot'),
        ('Box Plot', 'Box Plot'),
        ('Density Plot', 'Density Plot'),
    ]

    COLOR_THEMES = [
        ('viridis', 'Viridis'),
        ('plasma', 'Plasma'),
        ('inferno', 'Inferno'),
        ('magma', 'Magma'),
        ('Blues', 'Blues'),
        ('Reds', 'Reds'),
    ]

    plot_type = forms.ChoiceField(
        label="Тип графика",
        choices=PLOT_TYPES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    x_axis = forms.ChoiceField(
        label="Ось X",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    # y_axis не обязателен для Гистограммы, поэтому required=False
    y_axis = forms.ChoiceField(
        label="Ось Y",
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=False
    )
    hue = forms.ChoiceField(
        label="Группировка цветом (Hue)",
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=False
    )
    color_theme = forms.ChoiceField(
        label="Цветовая схема",
        choices=COLOR_THEMES,
        initial='viridis',
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    def __init__(self, *args, columns=None, **kwargs):
        super().__init__(*args, columns=columns, **kwargs)
        if columns:
            empty_choice = [('', '--- Не выбрано ---')]
            self.fields['hue'].choices = empty_choice + [(c, c) for c in columns]
            self.fields['y_axis'].choices = empty_choice + [(c, c) for c in columns]


# ==========================================
# 4. Форма обучения модели
# ==========================================

class TrainModelForm(BaseDynamicForm):
    TASK_CHOICES = [
        ('Regression', 'Регрессия'),
        ('Classification', 'Классификация'),
    ]

    ALGO_CHOICES = [
        # Регрессия
        ('Linear Regression', 'Linear Regression'),
        ('Decision Tree Regression', 'Decision Tree Regression'),
        ('Random Forest Regression', 'Random Forest Regression'),
        ('Support Vector Machine Regression', 'SVR'),
        ('Gradient Boosting Regression', 'Gradient Boosting Regression'),
        ('Multi Layer Perceptron Regression', 'MLP Regressor'),
        # Классификация
        ('Logistic Regression', 'Logistic Regression'),
        ('Decision Tree CLassifier', 'Decision Tree Classifier'),
        ('Random Forest Classification', 'Random Forest Classifier'),
        ('Support Vector Machine CLassification', 'SVC'),
        ('Gradient Boosting Classification', 'Gradient Boosting Classifier'),
        ('Multi Layer Perceptron Classifier', 'MLP Classifier'),
    ]

    target_column = forms.ChoiceField(
        label="Целевая переменная (y)",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    test_size = forms.FloatField(
        label="Размер тестовой выборки (0.1 - 0.9)",
        initial=0.2,
        min_value=0.01,
        max_value=0.99,
        # ИСПРАВЛЕНО: step='0.01' позволяет вводить 0.21, 0.33 и т.д.
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'})
    )

    task_type = forms.ChoiceField(
        label="Тип задачи",
        choices=TASK_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input', 'onclick': 'filterAlgorithms()'})
    )

    algorithm = forms.ChoiceField(
        label="Алгоритм",
        choices=ALGO_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'algo_select'})
    )

    # --- Гиперпараметры ---
    n_estimators = forms.IntegerField(
        label="n_estimators (Лес/Бустинг)",
        initial=100,
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control hyperparam-field'})
    )

    max_depth = forms.IntegerField(
        label="max_depth (Деревья)",
        initial=3,
        min_value=1,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control hyperparam-field'})
    )

    C_param = forms.FloatField(
        label="C (Логистическая/SVM)",
        initial=1.0,
        min_value=0.001,
        required=False,
        # ИСПРАВЛЕНО: Добавлен мелкий шаг
        widget=forms.NumberInput(attrs={'class': 'form-control hyperparam-field', 'step': '0.001'})
    )

    kernel = forms.ChoiceField(
        label="kernel (SVM)",
        choices=[('rbf', 'rbf'), ('linear', 'linear'), ('poly', 'poly')],
        initial='rbf',
        required=False,
        widget=forms.Select(attrs={'class': 'form-select hyperparam-field'})
    )

    learning_rate = forms.FloatField(
        label="learning_rate (Бустинг)",
        initial=0.1,
        min_value=0.0001,
        max_value=1.0,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control hyperparam-field', 'step': '0.0001'})
    )