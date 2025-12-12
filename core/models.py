import os
from django.db import models
from django.conf import settings


class UploadedDataset(models.Model):
    """
    Модель для хранения загруженных наборов данных.
    Привязана к сессии пользователя, чтобы анонимные пользователи могли работать со своими данными.
    """

    # Ключ сессии пользователя (из request.session.session_key)
    # unique=True гарантирует, что у одной сессии только один активный датасет
    session_key = models.CharField(
        max_length=40,
        unique=True,
        verbose_name="Ключ сессии"
    )

    # Оригинальный файл, загруженный пользователем (аналог st.session_state["original_data"])
    file = models.FileField(
        upload_to='datasets/original/',
        verbose_name="Оригинальный файл"
    )

    # Обработанный файл (после очистки, кодирования и т.д.) (аналог st.session_state["data"])
    # Может быть пустым, если изменений еще не было
    processed_file = models.FileField(
        upload_to='datasets/processed/',
        null=True,
        blank=True,
        verbose_name="Обработанный файл"
    )

    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Время загрузки")

    class Meta:
        verbose_name = "Загруженный датасет"
        verbose_name_plural = "Загруженные датасеты"

    def __str__(self):
        return f"Dataset for session {self.session_key} ({self.uploaded_at.strftime('%Y-%m-%d %H:%M')})"

    def get_active_file_path(self):
        """
        Возвращает путь к актуальному файлу.
        Если есть обработанная версия - возвращает её, иначе оригинал.
        """
        if self.processed_file:
            return self.processed_file.path
        return self.file.path

    def delete(self, *args, **kwargs):
        """
        Переопределение метода удаления, чтобы удалять файлы с диска
        при удалении записи из базы данных.
        """
        # Удаляем оригинал
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)

        # Удаляем обработанный файл
        if self.processed_file:
            if os.path.isfile(self.processed_file.path):
                os.remove(self.processed_file.path)

        super().delete(*args, **kwargs)