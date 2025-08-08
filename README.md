# PIRO Analyzer

**PIRO Analyzer** — это инструмент для анализа историко-этнографических травелогов с использованием PIRO-модели (People – Image – Representation – Occasion). Проект реализован на Python и предназначен для исследования репрезентаций этнических групп в путешественнической литературе (1850–1917 гг.).

## 📂 Структура

PIRO analyzer/
├── analysis/ # Модули анализа (темы, роли, сентимент и др.)
├── data/ # Входные файлы Excel (результаты PIRO-разметки)
├── logs/ # Логи выполнения скриптов
├── main.py # Основной скрипт запуска
├── requirements.txt # Зависимости проекта
├── README.md # Это описание


## 🚀 Установка

bash
git clone https://github.com/pavelaleks/piro-analyzer.git
cd piro-analyzer
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate для Windows
pip install -r requirements.txt

Использование
python analysis/analyze_to_excel.py

Скрипт прочитает файл data/results_piro_metada