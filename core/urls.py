from django.urls import path
from . import views

urlpatterns = [
    # Главная страница (страница приветствия)
    path('', views.home, name='home'),

    # Страница загрузки CSV файла
    path('upload/', views.upload_data, name='upload_data'),

    # Страница редактирования (удаление, пропуски, кодирование)
    path('edit/', views.edit_data, name='edit_data'),

    # Страница визуализации графиков
    path('visualize/', views.visualize_data, name='visualize_data'),

    # Страница обучения модели
    path('train/', views.train_model, name='train_model'),

    # Скрытый URL для скачивания файла (вызывается кнопкой)
    path('download_model/', views.download_model, name='download_model'),

    path('neural-network/', views.nn_builder, name='nn_builder'),
    path('api/train-nn/', views.api_train_nn, name='api_train_nn'),  # Скрытый API для AJAX
]