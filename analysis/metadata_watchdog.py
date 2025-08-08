import subprocess
import time
import logging
import sys
from pathlib import Path

# Логи для сторожа
LOG_DIR    = Path(__file__).resolve().parent / 'logs'
LOG_FILE   = LOG_DIR / 'metadata_watchdog.log'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Команда запуска: берём тот же интерпретатор, что и текущий процесс
METADATA_COMMAND = [sys.executable, '-u', '-m', 'analysis.metadata']

# Интервал между проверками (сек)
CHECK_PERIOD = 10

def start():
    logging.info("Starting metadata enrichment subprocess")
    proc = subprocess.Popen(
        METADATA_COMMAND,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1
    )
    return proc

def main():
    proc = start()
    while True:
        ret = proc.poll()
        if ret is None:
            # процесс ещё жив
            time.sleep(CHECK_PERIOD)
            continue

        if ret == 0:
            logging.info("Metadata enrichment finished successfully.")
            print("✅ Metadata enrichment completed with exit code 0.")
            break

        # аварийное завершение — перезапускаем
        logging.warning(f"Metadata process exited with code {ret}; restarting.")
        print(f"⚠️ Process exited ({ret}); restarting in 5s...")
        time.sleep(5)
        proc = start()

if __name__ == '__main__':
    main()
