import os
from core import init_application, build_processing_chain, process_folder, save_results

def run_console_app():
    """Запуск консольной версии приложения"""
    try:
        # Инициализация
        llm, parser, config, system_prompt = init_application()

        # Выбор режима обработки
        while True:
            mode_choice = input(
                "\nВыберите режим обработки:\n1. Объединённый режим (все изображения вместе)\n2. Разделённый режим (каждое изображение отдельно)\nВаш выбор (1/2): ").strip()
            if mode_choice in ('1', '2'):
                break
            else:
                print("Некорректный выбор. Попробуйте снова.")

        # Обработка папки с изображениями
        folder = input("\n📁 Введите путь к папке с UML-диаграммами: ")
        image_paths = process_folder(folder)

        if not image_paths:
            print("❌ В папке не найдены изображения с поддерживаемыми форматами (jpg, png, bmp)")
            exit()

        print(f"\n🔍 Найдено {len(image_paths)} изображений для обработки")

        # Создание цепи обработки
        chain = build_processing_chain(llm, parser, system_prompt)

        # Запуск конвертации
        print("\n🔄 Начинаю обработку UML-диаграмм...")

        if mode_choice == '1':  # Объединённый режим
            # Формируем строку с перечислением всех файлов
            filenames_comment = ", ".join(os.path.basename(path) for path in image_paths)
            comment_line = f"# Код для изображений: {filenames_comment}"

            result = chain.invoke(image_paths)
            # Добавляем строку с перечнем файлов в начало кода
            result.code = comment_line + "\n" + result.code
            save_results(result, config)

            if input("Показать полный код? (y/n): ").lower() == 'y':
                print("\n" + result.code)

        elif mode_choice == '2':  # Разделённый режим
            for img_path in image_paths:
                basename = os.path.splitext(os.path.basename(img_path))[0]
                print(f"\n🔎 Обрабатываю изображение: {basename}")

                result = chain.invoke([img_path])
                output_filename = f"{basename}_orm.py"

                # Добавляем имя изображения в качестве комментария в начале кода
                modified_code = f"# Код для изображения: {basename}\n" + result.code
                result.code = modified_code

                save_results(result, config, output_filename)

                if input(f"Показать код для {basename}? (y/n): ").lower() == 'y':
                    print("\n" + result.code)

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        if "credentials" in str(e).lower():
            print("Проверьте правильность ключа авторизации GigaChat")

if __name__ == "__main__":
    run_console_app()