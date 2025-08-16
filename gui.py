import os
import sys
import math
from typing import List, Tuple, Union, Dict, Any
from dotenv import find_dotenv, load_dotenv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QTextEdit, QPushButton, QRadioButton, QFileDialog,
                             QProgressBar, QMessageBox, QFormLayout, QComboBox, QFrame, QScrollArea,
                             QStackedLayout, QSizePolicy, QSplitter, QDesktopWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRectF, QSize, QPointF
from PyQt5.QtGui import (QFont, QTextCursor, QTextOption, QPixmap, QPainter, QWheelEvent,
                         QMouseEvent, QPaintEvent, QImage, QResizeEvent, QPalette, QColor)

# Импорт функционала из консольного приложения
from core import (
    DEFAULT_SYSTEM_PROMPT,
    ModelConfig,
    SQLAlchemyModels,
    build_processing_chain,
    process_folder,
    GigaChat,
    PydanticOutputParser
)


class ProcessingResult:
    """Класс для хранения результатов обработки с историей изменений"""

    def __init__(self, data: Union[SQLAlchemyModels, str], config: ModelConfig, image_name: str,
                 token_usage: int = 0, image_path: Union[str, List[str]] = None):
        self.data = data
        self.config = config
        self.image_name = image_name
        self.token_usage = token_usage
        self.is_error = isinstance(data, str)
        self.image_path = image_path  # Путь(и) к изображению(ям)

        # История изменений кода
        self.history: List[str] = []
        self.current_version = 0

        # Сохраняем оригинальный код
        if self.is_error:
            self.original_code = data
            self.history.append(data)
        else:
            self.original_code = data.code
            self.history.append(data.code)

    @property
    def code(self) -> str:
        """Текущая версия кода"""
        return self.history[self.current_version]

    @code.setter
    def code(self, value: str):
        """Обновление кода с сохранением в истории"""
        if len(self.history) == self.current_version + 1:
            self.history.append(value)
        else:
            # Если мы не на последней версии, перезаписываем историю
            self.history = self.history[:self.current_version + 1]
            self.history.append(value)
        self.current_version += 1

    @property
    def summary(self) -> str:
        if self.is_error:
            return f"Ошибка обработки изображения: {self.image_name}"
        return self.data.summary

    def restore_original(self):
        """Восстановление оригинального кода"""
        self.current_version = 0
        if self.is_error:
            return self.original_code
        return self.original_code

    def undo(self):
        """Восстановление предыдущей версии"""
        if self.current_version > 0:
            self.current_version -= 1
        return self.history[self.current_version]

    def redo(self):
        """Восстановление следующей версии"""
        if self.current_version < len(self.history) - 1:
            self.current_version += 1
        return self.history[self.current_version]

    def has_undo(self) -> bool:
        return self.current_version > 0

    def has_redo(self) -> bool:
        return self.current_version < len(self.history) - 1


class ImageViewer(QWidget):
    """Виджет для просмотра изображения с масштабированием и перемещением"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.drag_start = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, pixmap):
        self.pixmap = pixmap
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        if not self.pixmap or self.pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Масштабированное изображение
        scaled_pixmap = self.pixmap.scaled(
            self.pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Позиция для отрисовки
        x = (self.width() - scaled_pixmap.width()) // 2 + self.offset.x()
        y = (self.height() - scaled_pixmap.height()) // 2 + self.offset.y()

        # Заливка фона
        painter.fillRect(event.rect(), QColor(240, 240, 240))

        # Рисуем изображение
        painter.drawPixmap(x, y, scaled_pixmap)

        # Рисуем рамку
        painter.setPen(Qt.darkGray)
        painter.drawRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())

    def wheelEvent(self, event: QWheelEvent):
        """Масштабирование колесиком мыши"""
        zoom_in = event.angleDelta().y() > 0
        old_scale = self.scale_factor

        # Вычисляем точку относительно изображения до масштабирования
        pos_before = (event.position() - QPointF(self.rect().center() + self.offset)) / old_scale

        # Изменяем масштаб
        if zoom_in:
            self.scale_factor *= 1.1
        else:
            self.scale_factor *= 0.9

        # Ограничиваем масштаб
        self.scale_factor = max(0.1, min(self.scale_factor, 10.0))

        # Вычисляем новую позицию для сохранения точки под курсором
        pos_after = pos_before * self.scale_factor
        move = pos_after - pos_before

        # Корректируем смещение
        self.offset += move.toPoint()

        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drag_start:
            delta = event.pos() - self.drag_start
            self.offset += delta
            self.drag_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drag_start = None
            self.setCursor(Qt.ArrowCursor)

    def reset_view(self):
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.update()


class CollageWidget(QWidget):
    """Виджет для отображения коллажа из изображений"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = []
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_images(self, pixmaps):
        self.images = pixmaps
        self.update()

    def paintEvent(self, event: QPaintEvent):
        if not self.images:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Рассчитываем расположение изображений в сетке
        num_images = len(self.images)
        if num_images == 0:
            return

        # Определяем размер сетки
        if num_images <= 4:
            cols = 2
        else:
            cols = math.ceil(math.sqrt(num_images))

        rows = math.ceil(num_images / cols)

        cell_width = self.width() / cols
        cell_height = self.height() / rows

        # Заливка фона
        painter.fillRect(event.rect(), QColor(240, 240, 240))

        for i, pixmap in enumerate(self.images):
            if pixmap.isNull():
                continue

            row = i // cols
            col = i % cols

            # Масштабируем изображение под размер ячейки
            scaled = pixmap.scaled(
                int(cell_width - 10),
                int(cell_height - 10),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            x = col * cell_width + (cell_width - scaled.width()) / 2
            y = row * cell_height + (cell_height - scaled.height()) / 2

            # Рисуем рамку
            painter.setPen(Qt.darkGray)
            painter.drawRect(int(x), int(y), int(scaled.width()), int(scaled.height()))

            # Рисуем изображение
            painter.drawPixmap(int(x), int(y), scaled)


class ConversionThread(QThread):
    """Поток для выполнения конвертации"""
    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(object, object, str, int, object)  # Результат, конфиг, имя файла, токены, image_path
    error_occurred = pyqtSignal(str)

    def __init__(self, llm, parser, config, system_prompt, image_paths, mode, parent=None):
        super().__init__(parent)
        self.llm = llm
        self.parser = parser
        self.config = config
        self.system_prompt = system_prompt
        self.image_paths = image_paths
        self.mode = mode
        self.running = True

    def run(self):
        try:
            self.progress.emit(10, "Построение цепочки обработки...")
            chain = build_processing_chain(self.llm, self.parser, self.system_prompt)

            self.progress.emit(30, "Обработка изображений...")

            if self.mode == "combined":
                self.progress.emit(40, f"Обработка {len(self.image_paths)} изображений вместе")
                try:
                    # Получаем результат без метаданных
                    result = chain.invoke(self.image_paths)

                    # Добавляем комментарий с именами файлов
                    filenames = ", ".join(os.path.basename(p) for p in self.image_paths)
                    result.code = f"# Код для изображений: {filenames}\n" + result.code

                    # Пока токены недоступны
                    token_usage = 0
                    self.result_ready.emit(result, self.config, "combined", token_usage, self.image_paths)
                except Exception as e:
                    error_msg = f"Ошибка обработки объединённых изображений: {str(e)}"
                    self.result_ready.emit(error_msg, self.config, "combined", 0, self.image_paths)
            else:
                for i, img_path in enumerate(self.image_paths):
                    if not self.running:
                        return
                    basename = os.path.basename(img_path)
                    self.progress.emit(40 + int(i * 50 / len(self.image_paths)),
                                       f"Обработка изображения: {basename}")
                    try:
                        # Получаем результат без метаданных
                        result = chain.invoke([img_path])

                        # Добавляем комментарий с именем файла
                        result.code = f"# Код для изображения: {basename}\n" + result.code

                        # Пока токены недоступны
                        token_usage = 0
                        self.result_ready.emit(result, self.config, basename, token_usage, img_path)
                    except Exception as e:
                        error_msg = f"❌ Ошибка обработки {basename}: {str(e)}"
                        self.result_ready.emit(error_msg, self.config, basename, 0, img_path)

            self.progress.emit(100, "Обработка завершена!")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UML to SQLAlchemy Converter")

        # Установка начального размера окна
        self.setMinimumSize(800, 600)
        self.resize(1200, 900)

        # Инициализация состояния
        self.llm = None
        self.parser = None
        self.config = ModelConfig()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.conversion_thread = None
        self.results: List[ProcessingResult] = []
        self.current_result_index = -1
        self.output_dir = os.path.expanduser("~")  # Папка для сохранения по умолчанию
        self.editing_mode = False
        self.image_paths = []  # Пути к изображениям для текущего результата
        self.current_image_index = 0  # Текущий индекс изображения в слайдере
        self.results_tab_index = -1  # Индекс вкладки с результатами

        # Создание вкладок
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Создаем основные вкладки
        self.create_settings_tab()
        self.create_prompt_tab()
        self.create_processing_tab()
        self.create_results_tab()
        self.create_help_tab()  # Справка всегда последняя

        # Статус бар
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Инициализация приложения
        self.init_application()

        # Центрирование окна на экране
        self.center_window()

    def center_window(self):
        """Центрирует окно на экране"""
        frame = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

    def create_settings_tab(self):
        """Вкладка настроек модели"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Группа аутентификации
        auth_group = QGroupBox("Аутентификация")
        auth_layout = QVBoxLayout()

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("Введите ключ API")
        auth_layout.addWidget(QLabel("Ключ GigaChat API:"))
        auth_layout.addWidget(self.api_key)

        # Проверка переменных окружения
        if os.getenv("GIGACHAT_CREDENTIALS"):
            self.api_key.setPlaceholderText("Используется ключ из переменных окружения")
            self.api_key.setEnabled(False)
        else:
            self.api_key.setEnabled(True)

        auth_group.setLayout(auth_layout)
        layout.addWidget(auth_group)

        # Группа настроек модели
        model_group = QGroupBox("Настройки модели")
        model_layout = QFormLayout()

        # Выпадающий список для выбора модели
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"])
        self.model_combo.setCurrentIndex(1)  # По умолчанию Pro

        self.temperature = QLineEdit("0.1")
        self.max_tokens = QLineEdit("")
        self.timeout = QLineEdit("600")

        model_layout.addRow("Модель:", self.model_combo)
        model_layout.addRow("Температура (0-1):", self.temperature)
        model_layout.addRow("Макс. токенов (пусто=авто):", self.max_tokens)
        model_layout.addRow("Таймаут (сек):", self.timeout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Группа сохранения
        save_group = QGroupBox("Настройки сохранения")
        save_layout = QVBoxLayout()

        self.output_dir_label = QLabel(f"Папка для сохранения: {self.output_dir}")
        self.select_output_dir_btn = QPushButton("Выбрать папку для сохранения...")
        self.select_output_dir_btn.clicked.connect(self.select_output_directory)

        save_layout.addWidget(self.output_dir_label)
        save_layout.addWidget(self.select_output_dir_btn)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("Сохранить настройки")
        self.save_settings_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.save_settings_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Настройки")

    def create_prompt_tab(self):
        """Вкладка управления промптом"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Выбор источника промпта
        source_group = QGroupBox("Источник промпта")
        source_layout = QVBoxLayout()

        self.prompt_std_radio = QRadioButton("Стандартный промпт")
        self.prompt_std_radio.setChecked(True)
        self.prompt_file_radio = QRadioButton("Загрузить из файла")
        self.prompt_custom_radio = QRadioButton("Пользовательский промпт")

        source_layout.addWidget(self.prompt_std_radio)
        source_layout.addWidget(self.prompt_file_radio)
        source_layout.addWidget(self.prompt_custom_radio)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Поле для пользовательского промпта
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)
        self.prompt_edit.setReadOnly(True)
        self.prompt_edit.setMinimumHeight(300)
        layout.addWidget(QLabel("Текст промпта:"))
        layout.addWidget(self.prompt_edit)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.load_file_btn = QPushButton("Выбрать файл...")
        self.load_file_btn.clicked.connect(self.load_prompt_from_file)
        self.load_file_btn.setEnabled(False)

        self.save_prompt_btn = QPushButton("Применить промпт")
        self.save_prompt_btn.clicked.connect(self.apply_prompt)

        btn_layout.addWidget(self.load_file_btn)
        btn_layout.addWidget(self.save_prompt_btn)
        layout.addLayout(btn_layout)

        # Обработчики изменения выбора
        self.prompt_std_radio.toggled.connect(lambda: self.prompt_edit.setReadOnly(True))
        self.prompt_file_radio.toggled.connect(lambda: self.load_file_btn.setEnabled(True))
        self.prompt_custom_radio.toggled.connect(lambda: self.prompt_edit.setReadOnly(False))

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Промпт")

    def create_processing_tab(self):
        """Вкладка обработки изображений"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Выбор папки с изображениями
        folder_group = QGroupBox("Папка с изображениями")
        folder_layout = QHBoxLayout()

        self.folder_path = QLineEdit()
        self.folder_path.setReadOnly(True)
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.clicked.connect(self.select_folder)

        folder_layout.addWidget(self.folder_path)
        folder_layout.addWidget(self.browse_btn)
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        # Режим обработки
        mode_group = QGroupBox("Режим обработки")
        mode_layout = QVBoxLayout()

        self.mode_combined = QRadioButton("Объединённый режим (все изображения вместе)")
        self.mode_separate = QRadioButton("Разделённый режим (каждое изображение отдельно)")
        self.mode_combined.setChecked(True)

        mode_layout.addWidget(self.mode_combined)
        mode_layout.addWidget(self.mode_separate)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Информация о файлах
        self.file_info = QLabel("Файлы не выбраны")
        layout.addWidget(self.file_info)

        # Кнопки обработки
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Начать конвертацию")
        self.start_btn.clicked.connect(self.start_conversion)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Журнал обработки
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        self.log_view.setWordWrapMode(QTextOption.NoWrap)
        layout.addWidget(QLabel("Журнал обработки:"))
        layout.addWidget(self.log_view)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Обработка")

    def create_results_tab(self):
        """Создание и возврат вкладки результатов"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Панель навигации по результатам
        nav_frame = QFrame()
        nav_layout = QHBoxLayout(nav_frame)

        self.prev_result_btn = QPushButton("◄ Предыдущий")
        self.prev_result_btn.setEnabled(False)
        self.prev_result_btn.clicked.connect(self.show_previous_result)

        self.next_result_btn = QPushButton("Следующий ►")
        self.next_result_btn.setEnabled(False)
        self.next_result_btn.clicked.connect(self.show_next_result)

        self.result_counter = QLabel("0/0")
        self.result_counter.setFont(QFont("Arial", 10, QFont.Bold))

        nav_layout.addWidget(self.prev_result_btn)
        nav_layout.addWidget(self.result_counter)
        nav_layout.addWidget(self.next_result_btn)
        main_layout.addWidget(nav_frame)

        # Информация о результатах
        result_info_frame = QFrame()
        result_info_layout = QVBoxLayout(result_info_frame)

        self.results_info = QLabel("Результаты обработки будут отображены здесь")
        self.token_info = QLabel("Использовано токенов: 0")

        result_info_layout.addWidget(self.results_info)
        result_info_layout.addWidget(self.token_info)
        main_layout.addWidget(result_info_frame)

        # Основная область: разделитель между кодом и изображением
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # 1 означает, что виджет будет растягиваться

        # Левая панель: код и управление
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Просмотр и редактирование кода
        code_group = QGroupBox("Сгенерированный код")
        code_layout = QVBoxLayout(code_group)

        self.code_view = QTextEdit()
        self.code_view.setFontFamily("Courier")
        self.code_view.setReadOnly(True)
        self.code_view.setMinimumHeight(300)

        # Панель редактирования
        edit_frame = QFrame()
        edit_layout = QHBoxLayout(edit_frame)

        self.edit_btn = QPushButton("Редактировать")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self.toggle_edit_mode)

        self.save_edit_btn = QPushButton("Сохранить изменения")
        self.save_edit_btn.setEnabled(False)
        self.save_edit_btn.clicked.connect(self.save_edits)

        self.cancel_edit_btn = QPushButton("Отменить редактирование")
        self.cancel_edit_btn.setEnabled(False)
        self.cancel_edit_btn.clicked.connect(self.cancel_edits)

        self.restore_btn = QPushButton("Восстановить исходный")
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self.restore_original)

        # Кнопки истории изменений
        self.undo_btn = QPushButton("← Отменить")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.undo_changes)

        self.redo_btn = QPushButton("Повторить →")
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self.redo_changes)

        edit_layout.addWidget(self.edit_btn)
        edit_layout.addWidget(self.save_edit_btn)
        edit_layout.addWidget(self.cancel_edit_btn)
        edit_layout.addWidget(self.restore_btn)
        edit_layout.addWidget(self.undo_btn)
        edit_layout.addWidget(self.redo_btn)

        code_layout.addWidget(self.code_view)
        code_layout.addWidget(edit_frame)
        left_layout.addWidget(code_group)

        # Кнопки сохранения кода
        save_btn_layout = QHBoxLayout()
        self.save_code_btn = QPushButton("Сохранить текущий код")
        self.save_code_btn.setEnabled(False)
        self.save_code_btn.clicked.connect(self.save_code)

        self.save_all_btn = QPushButton("Сохранить все результаты")
        self.save_all_btn.setEnabled(False)
        self.save_all_btn.clicked.connect(self.save_all_results)

        save_btn_layout.addWidget(self.save_code_btn)
        save_btn_layout.addWidget(self.save_all_btn)
        left_layout.addLayout(save_btn_layout)

        splitter.addWidget(left_panel)

        # Правая панель: изображение и управление
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # --- ПАНЕЛЬ УПРАВЛЕНИЯ ПРОСМОТРОМ ИЗОБРАЖЕНИЙ ---
        view_control_group = QGroupBox("Управление изображением")
        view_control_layout = QVBoxLayout(view_control_group)

        # Верхняя строка: переключатель режимов
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Режим просмотра:"))

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Одиночное изображение", "Коллаж", "Слайдер"])
        self.view_mode_combo.currentIndexChanged.connect(self.switch_view_mode)
        mode_layout.addWidget(self.view_mode_combo, 1)  # Растягиваем комбобокс

        self.reset_view_btn = QPushButton("Сброс вида")
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.reset_view_btn.setVisible(False)
        mode_layout.addWidget(self.reset_view_btn)

        view_control_layout.addLayout(mode_layout)

        # Вторая строка: навигация для слайдера
        nav_layout = QHBoxLayout()
        self.prev_image_btn = QPushButton("← Предыдущее")
        self.prev_image_btn.clicked.connect(self.show_prev_image)
        self.prev_image_btn.setVisible(False)

        self.next_image_btn = QPushButton("Следующее →")
        self.next_image_btn.clicked.connect(self.show_next_image)
        self.next_image_btn.setVisible(False)

        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addWidget(self.next_image_btn)
        view_control_layout.addLayout(nav_layout)

        right_layout.addWidget(view_control_group)

        # --- ОБЛАСТЬ ПРОСМОТРА ИЗОБРАЖЕНИЙ ---
        image_group = QGroupBox("Исходные изображения")
        image_layout = QVBoxLayout(image_group)

        # Создаем стек для переключения между режимами просмотра
        self.view_stack = QStackedLayout()

        # Виджет для одиночного изображения
        self.image_viewer = ImageViewer()
        self.view_stack.addWidget(self.image_viewer)

        # Виджет для коллажа
        self.collage_widget = CollageWidget()
        self.view_stack.addWidget(self.collage_widget)

        # Виджет для слайдера
        self.slider_widget = QWidget()
        slider_layout = QVBoxLayout(self.slider_widget)
        self.slider_label = QLabel("Слайдер изображений")
        self.slider_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.slider_label)

        # Используем отдельный ImageViewer для слайдера
        self.slider_image_viewer = ImageViewer()
        slider_layout.addWidget(self.slider_image_viewer)

        self.view_stack.addWidget(self.slider_widget)

        # Обертка с прокруткой
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        container.setLayout(self.view_stack)
        scroll.setWidget(container)

        image_layout.addWidget(scroll)
        right_layout.addWidget(image_group, 1)  # Растягиваем изображение

        splitter.addWidget(right_panel)

        # Настройка размеров разделителя
        splitter.setSizes([600, 400])

        # Сохраняем индекс вкладки результатов
        self.results_tab_index = self.tabs.addTab(tab, "Результаты")
        return tab

    def create_help_tab(self):
        """Создание вкладки справки с улучшенным форматированием"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Создаем область прокрутки
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        # Заголовок
        title = QLabel("Справка по приложению")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin-bottom: 15px;")
        scroll_layout.addWidget(title)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #cccccc;")
        scroll_layout.addWidget(line)

        # Общее описание
        desc_group = QGroupBox("О приложении")
        desc_layout = QVBoxLayout()
        desc_text = QLabel(
            "<html>"
            "<b>Данное приложение автоматически преобразует рукописные UML-диаграммы в код ORM SQLAlchemy с использованием технологии искусственного интеллекта <a href=\"https://developers.sber.ru/portal/products/gigachat-api\">GigaChat</a>.</b><br/><br/>"
            "<b>Основные возможности:</b><br/>" 
            "<ul style=\"list-style-type: disc;\">"
            "<li>Преобразование рукописных схем в Python-код</li>"
            "<li>Гибкая настройка параметров генерации</li>"
            "<li>Редактирование и сохранение результатов</li>"
            "</ul><br/>"
            "<i>Разработчик:</i> Михаил Кузьмин"
            "</html>"
        )
        desc_text.setOpenExternalLinks(True)
        desc_text.setWordWrap(True)
        desc_text.setStyleSheet("line-height: 1.4;")
        desc_layout.addWidget(desc_text)
        desc_group.setLayout(desc_layout)
        scroll_layout.addWidget(desc_group)

        # Инструкция по использованию
        usage_group = QGroupBox("Инструкция по использованию")
        usage_layout = QVBoxLayout()
        usage_text = QLabel(
            "<b>1. Настройки</b><br>"
            "• Введите ключ API GigaChat<br>"
            "• Выберите модель<br>"
            "• Настройте параметры генерации<br><br>"

            "<b>2. Промпт</b><br>"
            "• Используйте стандартный или кастомный промпт<br>"
            "• Промпт определяет правила преобразования UML→SQLAlchemy<br><br>"

            "<b>3. Обработка</b><br>"
            "• Выберите папку с изображениями UML-диаграмм<br>"
            "• Выберите режим обработки:<br>"
            "&nbsp;&nbsp;- <b>Объединённый</b>: все изображения как единая система<br>"
            "&nbsp;&nbsp;- <b>Разделённый</b>: каждое изображение обрабатывается отдельно<br>"
            "• Нажмите <b>Начать конвертацию</b><br><br>"

            "<b>4. Результаты</b><br>"
            "• Просмотрите сгенерированный код<br>"
            "• Изучите исходные изображения<br>"
            "• Сохраните или отредактируйте результаты"
        )
        usage_text.setWordWrap(True)
        usage_text.setStyleSheet("line-height: 1.4;")
        usage_layout.addWidget(usage_text)
        usage_group.setLayout(usage_layout)
        scroll_layout.addWidget(usage_group)

        # Важная информация
        important_group = QGroupBox("❗ Важная информация")
        important_group.setStyleSheet("QGroupBox { color: #d9534f; }")
        important_layout = QVBoxLayout()

        warning_text = QLabel(
            "<b>Для работы с GigaChat API необходим сертификат НУЦ Минцифры!</b>"
        )
        warning_text.setStyleSheet("color: #d9534f; font-weight: bold;")
        important_layout.addWidget(warning_text)

        details_text = QLabel(
            "Без установки сертификата при выполнении запроса на получение токена доступа (POST /api/v2/oauth) "
            "будет возникать ошибка:<br>"
            "<code>[SSL: CERTIFICATE_VE Редактирование и сохранение результатов\nRIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain</code><br><br>"

            "<b>Решение:</b><br>"
            "1.Установите сертификат по инструкции: "
            "<a href='https://developers.sber.ru/docs/ru/gigachat/certificates'>"
            "https://developers.sber.ru/docs/ru/gigachat/certificates</a><br><br>"

            "При возникновении проблем с подключением к API GigaChat в первую очередь проверьте установку сертификатов."
        )
        details_text.setWordWrap(True)
        details_text.setOpenExternalLinks(True)
        details_text.setStyleSheet("line-height: 1.4;")
        important_layout.addWidget(details_text)
        important_group.setLayout(important_layout)
        scroll_layout.addWidget(important_group)

        # Ссылки
        links_group = QGroupBox("🔗 Полезные ссылки")
        links_layout = QVBoxLayout()

        github_link = QLabel(
            "• <b>GitHub проекта</b>: "
            "<a href='https://github.com/mikhail-ku/uml-to-sqlalchemy-converter'>"
            "github.com/mikhail-ku/uml-to-sqlalchemy-converter</a>"
        )
        github_link.setOpenExternalLinks(True)

        gigachat_docs = QLabel(
            "• <b>Документация GigaChat API</b>: "
            "<a href='https://developers.sber.ru/docs/ru/gigachat/api/overview'>"
            "https://developers.sber.ru/docs/ru/gigachat/api/overview</a>"
        )
        gigachat_docs.setOpenExternalLinks(True)

        issues = QLabel(
            "• <b>Сообщить о проблеме</b>: "
            "<a href='https://github.com/mikhail-ku/uml-to-sqlalchemy-converter/issues'>"
            "Создать Issue на GitHub</a>"
        )
        issues.setOpenExternalLinks(True)

        links_layout.addWidget(github_link)
        links_layout.addWidget(gigachat_docs)
        links_layout.addWidget(issues)
        links_group.setLayout(links_layout)
        scroll_layout.addWidget(links_group)

        # Завершающий текст
        footer = QLabel(
            "© 2025 Михаил Кузьмин | Версия приложения 4.2.2<br>"
            "Для сообщений об ошибках и предложений используйте Issues на GitHub"
        )
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #777777; font-size: 10px; margin-top: 20px;")
        scroll_layout.addWidget(footer)

        # Растягивающий элемент
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Справка")
        return tab

    def init_application(self):
        """Инициализация приложения"""
        # Загрузка переменных окружения
        load_dotenv(find_dotenv())

        # Обновление UI
        self.temperature.setText(str(self.config.temperature))
        self.max_tokens.setText(str(self.config.max_tokens) if self.config.max_tokens else "")
        self.timeout.setText(str(self.config.timeout))

        self.log_message("Приложение инициализировано. Проверьте настройки перед запуском.")

    def save_settings(self):
        """Сохранение настроек модели"""
        try:
            # Получаем выбранную модель из выпадающего списка
            self.config.name = self.model_combo.currentText()
            self.config.temperature = float(self.temperature.text())
            self.config.max_tokens = int(self.max_tokens.text()) if self.max_tokens.text() else None
            self.config.timeout = int(self.timeout.text())

            # Обновление ключа API
            api_key = self.api_key.text()
            if api_key and not os.getenv("GIGACHAT_CREDENTIALS"):
                os.environ["GIGACHAT_CREDENTIALS"] = api_key

            self.log_message("Настройки успешно сохранены!")
        except Exception as e:
            self.log_message(f"Ошибка сохранения настроек: {str(e)}", "error")

    def apply_prompt(self):
        """Применение выбранного промпта"""
        try:
            if self.prompt_std_radio.isChecked():
                self.system_prompt = DEFAULT_SYSTEM_PROMPT
                self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)
            elif self.prompt_custom_radio.isChecked():
                self.system_prompt = self.prompt_edit.toPlainText()

            # Проверка плейсхолдера
            if "{format_instructions}" not in self.system_prompt:
                self.system_prompt += "\n\n{format_instructions}"
                self.prompt_edit.append("\n\n{format_instructions}")
                self.log_message("⚠️ В промпт добавлен плейсхолдер {format_instructions}")

            self.log_message("Промпт успешно применён!")
        except Exception as e:
            self.log_message(f"Ошибка применения промпта: {str(e)}", "error")

    def load_prompt_from_file(self):
        """Загрузка промпта из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с промптом", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.prompt_edit.setPlainText(content)
                    self.log_message(f"Промпт загружен из файла: {os.path.basename(file_path)}")
            except Exception as e:
                self.log_message(f"Ошибка загрузки промпта: {str(e)}", "error")

    def select_folder(self):
        """Выбор папки с изображениями"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с UML-диаграммами")

        if folder:
            self.folder_path.setText(folder)
            try:
                image_paths = process_folder(folder)

                if not image_paths:
                    self.file_info.setText("❌ В папке не найдены изображения с поддерживаемыми форматами")
                    self.start_btn.setEnabled(False)
                else:
                    self.file_info.setText(f"✅ Найдено {len(image_paths)} изображений: " +
                                           ", ".join(os.path.basename(p) for p in image_paths))
                    self.start_btn.setEnabled(True)
            except Exception as e:
                self.log_message(f"Ошибка при сканировании папки: {str(e)}", "error")

    def select_output_directory(self):
        """Выбор папки для сохранения результатов"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if folder:
            self.output_dir = folder
            self.output_dir_label.setText(f"Папка для сохранения: {self.output_dir}")
            self.log_message(f"Папка для сохранения установлена: {self.output_dir}")

    def start_conversion(self):
        """Запуск процесса конвертации"""
        # Остановка предыдущего потока, если он работает
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.conversion_thread.stop()
            self.conversion_thread.wait(2000)
            self.log_message("Предыдущая обработка остановлена")

        if not self.llm or not self.parser:
            try:
                # Инициализация GigaChat
                self.llm = GigaChat(
                    model=self.config.name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    verify_ssl_certs=False,
                    timeout=self.config.timeout,
                    credentials=os.getenv("GIGACHAT_CREDENTIALS")
                )

                # Парсер для структурированного вывода
                self.parser = PydanticOutputParser(pydantic_object=SQLAlchemyModels)
                self.log_message("Модель GigaChat успешно инициализирована")
            except Exception as e:
                self.log_message(f"Ошибка инициализации модели: {str(e)}", "error")
                return

        folder = self.folder_path.text()
        if not folder:
            self.log_message("❌ Папка с изображениями не выбрана", "error")
            return

        try:
            image_paths = process_folder(folder)
            if not image_paths:
                self.log_message("❌ В папке не найдены изображения с поддерживаемыми форматами", "error")
                return
        except Exception as e:
            self.log_message(f"Ошибка при сканировании папки: {str(e)}", "error")
            return

        mode = "combined" if self.mode_combined.isChecked() else "separate"

        # Сброс предыдущих результатов
        self.results = []
        self.current_result_index = -1
        self.code_view.clear()
        self.save_code_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        self.edit_btn.setEnabled(False)
        self.restore_btn.setEnabled(False)
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)

        # Сброс изображений
        self.image_paths = []
        self.current_image_index = 0
        self.image_viewer.set_image(QPixmap())
        self.collage_widget.set_images([])
        self.slider_image_viewer.set_image(QPixmap())
        self.slider_label.setText("Слайдер изображений")

        # Обновление навигации
        self.update_navigation_buttons()

        # Настройка интерфейса
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Переключаемся на вкладку результатов
        self.tabs.setCurrentIndex(self.results_tab_index)

        # Запуск потока обработки
        self.conversion_thread = ConversionThread(
            self.llm, self.parser, self.config, self.system_prompt, image_paths, mode
        )
        self.conversion_thread.progress.connect(self.update_progress)
        self.conversion_thread.result_ready.connect(self.handle_result)
        self.conversion_thread.error_occurred.connect(self.handle_error)
        self.conversion_thread.finished.connect(self.conversion_finished)
        self.conversion_thread.start()

        self.log_message("🚀 Запуск обработки изображений...")

    def stop_conversion(self):
        """Остановка процесса конвертации"""
        if self.conversion_thread:
            self.conversion_thread.stop()
            if self.conversion_thread.isRunning():
                self.conversion_thread.wait(2000)
            self.log_message("Обработка остановлена пользователем")
            self.conversion_finished()

    def update_progress(self, value, message):
        """Обновление прогресса и сообщений"""
        self.progress_bar.setValue(value)
        self.log_message(message)
        self.status_bar.showMessage(message)

    def handle_result(self, result, config, image_name, token_usage, image_path):
        """Обработка результата конвертации"""
        # Создаем объект результата
        processing_result = ProcessingResult(result, config, image_name, token_usage, image_path)
        self.results.append(processing_result)

        # Если это первый результат, показываем его
        if len(self.results) == 1:
            self.current_result_index = 0
            self.display_current_result()

        # Обновляем навигацию
        self.update_navigation_buttons()

        # Активируем кнопку сохранения всех результатов
        if len(self.results) > 0:
            self.save_all_btn.setEnabled(True)

        # Логируем результат
        if processing_result.is_error:
            self.log_message(f"❌ Ошибка обработки: {image_name}", "error")
        else:
            self.log_message(f"✅ Успешно обработано: {image_name}")

    def display_current_result(self):
        """Отображение текущего результата"""
        if not self.results or self.current_result_index < 0:
            return

        result = self.results[self.current_result_index]

        # Отображаем информацию о результате
        self.results_info.setText(result.summary)
        self.token_info.setText(f"Использовано токенов: {result.token_usage}")

        # Отображаем код или сообщение об ошибке
        self.code_view.setPlainText(result.code)

        # Прокручиваем в начало
        self.code_view.moveCursor(QTextCursor.Start)

        # Активируем кнопки
        self.save_code_btn.setEnabled(True)
        self.edit_btn.setEnabled(True)
        self.restore_btn.setEnabled(True)
        self.undo_btn.setEnabled(result.has_undo())
        self.redo_btn.setEnabled(result.has_redo())

        # Обновляем счётчик
        self.result_counter.setText(f"{self.current_result_index + 1}/{len(self.results)}")

        # Выходим из режима редактирования
        self.cancel_edits()

        # Обновляем изображения
        self.update_image_display(result)

    def update_image_display(self, result: ProcessingResult):
        """Обновление отображения изображений для текущего результата"""
        # Получаем пути к изображениям
        if result.image_path:
            if isinstance(result.image_path, list):
                self.image_paths = result.image_path
            else:
                self.image_paths = [result.image_path]
        else:
            self.image_paths = []

        # Сбрасываем индекс изображения для слайдера
        self.current_image_index = 0

        # Обновляем отображение в соответствии с текущим режимом
        self.switch_view_mode(self.view_mode_combo.currentIndex())

        # Если нет изображений, очищаем просмотрщики
        if not self.image_paths:
            self.image_viewer.set_image(QPixmap())
            self.collage_widget.set_images([])
            self.slider_image_viewer.set_image(QPixmap())
            self.slider_label.setText("Изображения недоступны")

    def switch_view_mode(self, index):
        """Переключение между режимами просмотра"""
        if index >= self.view_stack.count():
            return

        self.view_stack.setCurrentIndex(index)

        # Показываем/скрываем кнопки управления
        is_slider_mode = (index == 2)
        self.prev_image_btn.setVisible(is_slider_mode)
        self.next_image_btn.setVisible(is_slider_mode)
        self.reset_view_btn.setVisible(index != 1)  # Для коллажа не нужен сброс

        # Обновляем отображение
        if not self.image_paths:
            return

        if index == 0:  # Одиночное изображение
            if self.image_paths:
                try:
                    pixmap = QPixmap(self.image_paths[0])
                    self.image_viewer.set_image(pixmap)
                except Exception as e:
                    self.log_message(f"Ошибка загрузки изображения: {str(e)}", "error")
            else:
                self.image_viewer.set_image(QPixmap())

        elif index == 1:  # Коллаж
            pixmaps = []
            for path in self.image_paths:
                try:
                    pixmap = QPixmap(path)
                    if not pixmap.isNull():
                        pixmaps.append(pixmap)
                except Exception as e:
                    self.log_message(f"Ошибка загрузки изображения {path}: {str(e)}", "error")
            self.collage_widget.set_images(pixmaps)

        elif index == 2:  # Слайдер
            self.show_image_at_index(self.current_image_index)

    def show_image_at_index(self, index):
        """Показать изображение по индексу в слайдере"""
        if not self.image_paths or index < 0 or index >= len(self.image_paths):
            return

        self.current_image_index = index
        try:
            pixmap = QPixmap(self.image_paths[index])
            self.slider_image_viewer.set_image(pixmap)

            # Обновляем информацию
            self.slider_label.setText(
                f"Изображение {index + 1} из {len(self.image_paths)}: {os.path.basename(self.image_paths[index])}"
            )

            # Обновляем состояние кнопок навигации
            self.prev_image_btn.setEnabled(index > 0)
            self.next_image_btn.setEnabled(index < len(self.image_paths) - 1)
        except Exception as e:
            self.log_message(f"Ошибка загрузки изображения: {str(e)}", "error")

    def show_prev_image(self):
        """Показать предыдущее изображение в слайдере"""
        if self.current_image_index > 0:
            self.show_image_at_index(self.current_image_index - 1)

    def show_next_image(self):
        """Показать следующее изображение в слайдере"""
        if self.current_image_index < len(self.image_paths) - 1:
            self.show_image_at_index(self.current_image_index + 1)

    def reset_view(self):
        """Сброс масштаба и положения изображения"""
        current_index = self.view_stack.currentIndex()

        if current_index == 0:  # Одиночное изображение
            self.image_viewer.reset_view()
        elif current_index == 2:  # Слайдер
            self.slider_image_viewer.reset_view()

    def update_navigation_buttons(self):
        """Обновление состояния кнопок навигации"""
        has_results = len(self.results) > 0
        self.prev_result_btn.setEnabled(has_results and self.current_result_index > 0)
        self.next_result_btn.setEnabled(has_results and self.current_result_index < len(self.results) - 1)
        self.result_counter.setEnabled(has_results)

        if not has_results:
            self.result_counter.setText("0/0")
        else:
            self.result_counter.setText(f"{self.current_result_index + 1}/{len(self.results)}")

    def show_previous_result(self):
        """Показать предыдущий результат"""
        if self.current_result_index > 0:
            self.current_result_index -= 1
            self.display_current_result()
            self.update_navigation_buttons()

    def show_next_result(self):
        """Показать следующий результат"""
        if self.current_result_index < len(self.results) - 1:
            self.current_result_index += 1
            self.display_current_result()
            self.update_navigation_buttons()

    def toggle_edit_mode(self):
        """Переключение режима редактирования"""
        if self.code_view.isReadOnly():
            self.code_view.setReadOnly(False)
            self.edit_btn.setEnabled(False)
            self.save_edit_btn.setEnabled(True)
            self.cancel_edit_btn.setEnabled(True)
            self.restore_btn.setEnabled(False)
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            self.log_message("Режим редактирования включен. Измените код по необходимости.")
        else:
            self.code_view.setReadOnly(True)
            self.edit_btn.setEnabled(True)
            self.save_edit_btn.setEnabled(False)
            self.cancel_edit_btn.setEnabled(False)
            self.restore_btn.setEnabled(True)
            self.undo_btn.setEnabled(self.results[self.current_result_index].has_undo())
            self.redo_btn.setEnabled(self.results[self.current_result_index].has_redo())

    def save_edits(self):
        """Сохранение изменений в текущем результате"""
        if self.current_result_index < 0 or not self.results:
            return

        # Обновляем код в текущем результате
        new_code = self.code_view.toPlainText()
        result = self.results[self.current_result_index]
        result.code = new_code

        self.toggle_edit_mode()
        self.log_message("Изменения сохранены в текущем результате.")
        self.undo_btn.setEnabled(result.has_undo())
        self.redo_btn.setEnabled(result.has_redo())

    def cancel_edits(self):
        """Отмена изменений и возврат к исходному результату"""
        self.code_view.setReadOnly(True)
        self.edit_btn.setEnabled(True)
        self.save_edit_btn.setEnabled(False)
        self.cancel_edit_btn.setEnabled(False)
        self.restore_btn.setEnabled(True)

        # Восстанавливаем текущий текст
        if self.current_result_index >= 0 and self.results:
            result = self.results[self.current_result_index]
            self.code_view.setPlainText(result.code)
            self.code_view.moveCursor(QTextCursor.Start)
            self.undo_btn.setEnabled(result.has_undo())
            self.redo_btn.setEnabled(result.has_redo())

    def restore_original(self):
        """Восстановление оригинального кода"""
        if self.current_result_index < 0 or not self.results:
            return

        result = self.results[self.current_result_index]
        result.restore_original()
        self.code_view.setPlainText(result.code)
        self.code_view.moveCursor(QTextCursor.Start)
        self.log_message("Исходный код восстановлен")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)

    def undo_changes(self):
        """Отмена последнего изменения"""
        if self.current_result_index < 0 or not self.results:
            return

        result = self.results[self.current_result_index]
        code = result.undo()
        self.code_view.setPlainText(code)
        self.undo_btn.setEnabled(result.has_undo())
        self.redo_btn.setEnabled(result.has_redo())
        self.log_message("Отменено последнее изменение")

    def redo_changes(self):
        """Повтор последнего изменения"""
        if self.current_result_index < 0 or not self.results:
            return

        result = self.results[self.current_result_index]
        code = result.redo()
        self.code_view.setPlainText(code)
        self.undo_btn.setEnabled(result.has_undo())
        self.redo_btn.setEnabled(result.has_redo())
        self.log_message("Повторено последнее изменение")

    def save_code(self):
        """Сохранение текущего результата в файл"""
        if not self.results or self.current_result_index < 0:
            return

        result = self.results[self.current_result_index]

        # Предлагаем выбрать имя файла
        default_name = f"models_{result.config.name.replace('-', '_')}.py"
        if result.image_name != "combined":
            default_name = f"{os.path.splitext(result.image_name)[0]}_orm.py"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить код", os.path.join(self.output_dir, default_name),
            "Python Files (*.py);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if result.is_error:
                    f.write(result.code)
                else:
                    f.write("# " + result.summary.strip().replace('\n', '\n# ') + "\n\n")
                    f.write(result.code)

            self.log_message(f"✅ Код успешно сохранён в {file_path}")
        except Exception as e:
            self.log_message(f"❌ Ошибка сохранения кода: {str(e)}", "error")

    def save_all_results(self):
        """Сохранение всех результатов в выбранную папку"""
        if not self.results:
            return

        # Создаем подпапку для результатов
        save_dir = os.path.join(self.output_dir, "sqlalchemy_models")
        os.makedirs(save_dir, exist_ok=True)

        success_count = 0
        error_count = 0

        for result in self.results:
            try:
                if result.image_name == "combined":
                    filename = "combined_models.py"
                else:
                    filename = f"{os.path.splitext(result.image_name)[0]}_orm.py"

                file_path = os.path.join(save_dir, filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    if result.is_error:
                        f.write(result.code)
                        error_count += 1
                    else:
                        f.write("# " + result.summary.strip().replace('\n', '\n# ') + "\n\n")
                        f.write(result.code)
                        success_count += 1
            except Exception as e:
                error_count += 1
                self.log_message(f"❌ Ошибка сохранения {filename}: {str(e)}", "error")

        self.log_message(f"✅ Все результаты сохранены в папку: {save_dir}")
        self.log_message(f"Успешно: {success_count}, Ошибок: {error_count}")

    def handle_error(self, error):
        """Обработка ошибки конвертации"""
        self.log_message(f"❌ Ошибка обработки: {error}", "error")
        if "credentials" in error.lower():
            self.log_message("Проверьте правильность ключа авторизации GigaChat", "error")

    def conversion_finished(self):
        """Завершение процесса конвертации"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.log_message("Обработка завершена")

        if not self.results:
            self.log_message("Нет результатов для отображения", "warning")
        else:
            # Показываем первый результат, если ещё не показали
            if self.current_result_index < 0:
                self.current_result_index = 0
                self.display_current_result()

            self.update_navigation_buttons()

    def log_message(self, message, level="info"):
        """Добавление сообщения в журнал"""
        if level == "error":
            formatted = f'<span style="color:red;">❌ {message}</span>'
        elif level == "warning":
            formatted = f'<span style="color:orange;">⚠️ {message}</span>'
        else:
            # Проверяем наличие эмодзи в сообщении для определения цвета
            if "✅" in message:
                formatted = f'<span style="color:green;">{message}</span>'
            elif "❌" in message:
                formatted = f'<span style="color:red;">{message}</span>'
            else:
                formatted = message

        self.log_view.append(formatted)
        # Автоматическая прокрутка вниз
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.conversion_thread.stop()
            if self.conversion_thread.isRunning():
                self.conversion_thread.wait(2000)
        event.accept()


def run_gui():
    """Запуск графического интерфейса"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()