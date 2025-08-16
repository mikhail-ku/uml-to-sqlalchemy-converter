import os
import getpass
import re
from typing import List, Optional
from dotenv import find_dotenv, load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

# Загрузка переменных окружения
load_dotenv(find_dotenv())

# Чтение системного промпта из внешнего файла
with open("prompts/system_prompt.txt", encoding="utf-8") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()

# --- КОНФИГУРАЦИЯ МОДЕЛИ ---
class ModelConfig(BaseModel):
    """Конфигурация модели GigaChat"""
    name: str = Field("GigaChat-2-Pro", description="Название модели")
    temperature: float = Field(0.1, description="Творчество ответов (0-1)")
    max_tokens: Optional[int] = Field(None, description="Макс. токенов в ответе")
    verify_ssl: bool = Field(False, description="Проверять SSL сертификаты")
    timeout: int = Field(600, description="Таймаут запроса в секундах")


# --- СТРУКТУРА ОТВЕТА ---
class SQLAlchemyModels(BaseModel):
    """Сгенерированные ORM-модели SQLAlchemy"""
    code: str = Field(..., description="Python код с ORM-моделями SQLAlchemy 2.x")
    summary: str = Field(..., description="Краткое описание моделей и отношений")
    token_usage: int = Field(0, description="Использовано токенов")

    @validator('code')
    def code_must_contain_models(cls, v):
        if 'declarative_base' not in v:
            raise ValueError('Сгенерированный код не содержит базовый класс SQLAlchemy')
        return v


# --- ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ---
def init_application():
    """Инициализация приложения с интерактивным конфигурированием"""
    print("🔄 Инициализация конвертера UML в SQLAlchemy ORM")
    print("--------------------------------------------------")

    # Конфигурация модели
    config = ModelConfig()
    system_prompt = DEFAULT_SYSTEM_PROMPT

    if input("Изменить настройки модели? (y/n): ").lower() == 'y':
        config.name = input(f"Модель ({config.name}): ") or config.name
        config.temperature = float(input(f"Temperature ({config.temperature}): ") or config.temperature)
        config.max_tokens = int(input(f"Макс. токенов ({config.max_tokens}): ") or 0) or None
        config.timeout = int(input(f"Таймаут ({config.timeout}): ") or config.timeout)

    # Настройка промпта
    if input("Использовать кастомный промпт? (y/n): ").lower() == 'y':
        prompt_choice = input("1. Ввести путь к файлу\n2. Ввести текст промпта\nВыбор (1/2): ")

        if prompt_choice == '1':
            file_path = input("Путь к файлу с промптом: ")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
                print("✅ Промпт успешно загружен")
            except Exception as e:
                print(f"❌ Ошибка загрузки промпта: {e}. Используется стандартный промпт")
        elif prompt_choice == '2':
            print("Введите текст промпта (Ctrl+D/Ctrl+Z для завершения):")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                lines.append(line)
            system_prompt = "\n".join(lines)
            print("✅ Промпт успешно принят")
        else:
            print("❌ Неверный выбор. Используется стандартный промпт")

    # Проверка наличия плейсхолдера в промпте
    if "{format_instructions}" not in system_prompt:
        system_prompt += "\n\n{format_instructions}"
        print("⚠️ В промпт добавлен плейсхолдер {format_instructions}")

    # Инициализация GigaChat
    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("🔑 Введите ключ авторизации GigaChat API: ")

    llm = GigaChat(
        model=config.name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verify_ssl_certs=config.verify_ssl,
        timeout=config.timeout
    )

    # Парсер для структурированного вывода
    parser = PydanticOutputParser(pydantic_object=SQLAlchemyModels)

    return llm, parser, config, system_prompt


# --- ОСНОВНАЯ ЛОГИКА ---
def build_processing_chain(llm, parser, system_prompt):
    """Строит цепочку обработки изображений"""

    # Функция для загрузки файлов
    def upload_files(file_paths: List[str]) -> List[str]:
        file_ids = []
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    uploaded_file = llm.upload_file(f)
                    file_ids.append(uploaded_file.id_)
                    print(f"✅ Загружено: {os.path.basename(path)}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {path}: {str(e)}")
        return file_ids

    # Динамические инструкции парсера
    format_instructions = parser.get_format_instructions()

    # Сборка цепочки обработки
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history"),
    ]).partial(format_instructions=format_instructions)

    return (
            RunnablePassthrough()
            | RunnableLambda(upload_files)
            | RunnableLambda(lambda ids: {
        "history": [HumanMessage(content="", additional_kwargs={"attachments": ids})]
    })
            | prompt
            | llm
            | parser
    )


# --- ОБРАБОТКА ПАПКИ С ИЗОБРАЖЕНИЯМИ ---
def process_folder(folder_path: str) -> List[str]:
    """Возвращает список UML-изображений в папке"""
    if not os.path.isdir(folder_path):
        raise ValueError(f"❌ Папка не существует: {folder_path}")

    supported_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]


# --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
def save_results(models: SQLAlchemyModels, config: ModelConfig, output_filename=None):
    """Сохраняет результаты и выводит статистику"""
    if not output_filename:
        output_filename = f"models_{config.name.replace('-', '_')}.py"

    with open(output_filename, "w") as f:
        f.write("# " + models.summary.strip().replace('\n', '\n# ') + "\n\n")
        f.write(models.code)

    print("\n" + "=" * 50)
    print(f"💾 Сгенерированные модели сохранены в {output_filename}")
    print(f"🔍 Сводка: {models.summary}")
    print(f"🧮 Использовано токенов: {models.token_usage}")
    print("=" * 50)