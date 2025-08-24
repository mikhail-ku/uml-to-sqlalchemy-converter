#!/usr/bin/env python3
"""
Простой валидатор SQLAlchemy кода
Проверяет синтаксис и базовую валидность сгенерированного кода
"""

import ast


def validate_sqlalchemy_code(code):
    """
    Проверяет валидность SQLAlchemy кода
    Возвращает словарь с результатами проверки
    """
    results = {
        'syntax_valid': False,
        'base_class_found': False,
        'columns_found': False,
        'relationships_found': False,
        'errors': [],
        'warnings': []
    }

    # 1. Проверка синтаксиса Python
    try:
        ast.parse(code)
        results['syntax_valid'] = True
        print("✅ Синтаксис Python корректен")
    except SyntaxError as e:
        error_msg = f"❌ Синтаксическая ошибка: {e.msg} (строка {e.lineno})"
        results['errors'].append(error_msg)
        print(error_msg)
        return results

    # 2. Проверка базовых элементов SQLAlchemy
    if 'declarative_base' in code:
        results['base_class_found'] = True
        print("✅ Базовый класс моделей найден")
    else:
        error_msg = "❌ Не найден declarative_base - базовый класс моделей"
        results['errors'].append(error_msg)
        print(error_msg)

    if 'Column(' in code:
        results['columns_found'] = True
        print("✅ Определения колонок найдены")
    else:
        warning_msg = "⚠️  Не найдены определения колонок (Column)"
        results['warnings'].append(warning_msg)
        print(warning_msg)

    if 'relationship(' in code:
        results['relationships_found'] = True
        print("✅ Определения отношений найдены")
    else:
        warning_msg = "⚠️  Не найдены определения отношений (relationship)"
        results['warnings'].append(warning_msg)
        print(warning_msg)

    # 3. Проверка необходимых импортов
    required_imports = ['sqlalchemy', 'Column', 'Integer', 'String', 'ForeignKey']
    missing_imports = []

    for imp in required_imports:
        if imp not in code:
            missing_imports.append(imp)

    if missing_imports:
        warning_msg = f"⚠️  Отсутствуют импорты: {', '.join(missing_imports)}"
        results['warnings'].append(warning_msg)
        print(warning_msg)
    else:
        print("✅ Все необходимые импорты найдены")

    return results


def main():
    print("=" * 60)
    print("ВАЛИДАТОР SQLAlchemy КОДА")
    print("=" * 60)
    print("Введите код SQLAlchemy (Ctrl+D или пустая строка для завершения):")
    print("-" * 60)

    # Чтение многострочного ввода
    code_lines = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                break
            code_lines.append(line)
    except EOFError:
        pass

    if not code_lines:
        print("❌ Код не введен!")
        return

    code = "\n".join(code_lines)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print("=" * 60)

    # Проверка кода
    results = validate_sqlalchemy_code(code)

    # Общая оценка
    print("\n" + "=" * 60)
    if results['syntax_valid'] and results['base_class_found']:
        if not results['errors']:
            print("✅ КОД ВАЛИДЕН: может быть использован в проекте")
        else:
            print("⚠️  КОД ТРЕБУЕТ ДОРАБОТКИ: есть ошибки, но основа корректна")
    else:
        print("❌ КОД НЕВАЛИДЕН: требует значительной доработки")
    print("=" * 60)


if __name__ == "__main__":
    main()