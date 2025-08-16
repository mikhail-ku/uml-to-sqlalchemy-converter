import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='UML to SQLAlchemy Converter')
    parser.add_argument('--console', action='store_true', help='Run in console mode')
    args = parser.parse_args()

    if args.console:
        from console_app import run_console_app
        run_console_app()
    else:
        from gui import run_gui
        run_gui()

if __name__ == "__main__":
    main()