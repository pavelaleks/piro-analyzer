import logging
import pandas as pd
from pathlib import Path

# === НАСТРОЙКИ ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
INPUT_FILE = DATA_DIR / "results_piro_metadata.xlsx"
OUTPUT_FILE = DATA_DIR / "pirotable_analytics.xlsx"
LOG_FILE = LOGS_DIR / "analyze_to_excel.log"

# === ЛОГИРОВАНИЕ ===
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def run_analysis():
    try:
        logging.info("📥 Reading data from Excel: %s", INPUT_FILE)
        df = pd.read_excel(INPUT_FILE)

        import analysis.authors as authors
        import analysis.frequency as frequency
        import analysis.roles as roles
        import analysis.sentiment as sentiment
        import analysis.topics as topics
        import analysis.trends as trends
        import analysis.summarize as summarize
        import analysis.comention_network as comention

        logging.info("🔍 Starting analysis modules...")

        results = {
            "Summary Stats": summarize.analyze(df),
            "Sentiment Summary": sentiment.analyze(df),
            "Topic Frequency": topics.analyze(df),
            "Ethnic Roles": roles.analyze(df),
            "Trends Over Time": trends.analyze(df),
            "Co-Mention Network": comention.analyze(df),
            "Author Stats": authors.analyze(df),
            "Mention Frequency": frequency.analyze(df),
        }

        logging.info("📤 Writing results to Excel: %s", OUTPUT_FILE)
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            for sheet, result_df in results.items():
                if isinstance(result_df, pd.DataFrame):
                    result_df.to_excel(writer, sheet_name=sheet[:31], index=False)
                    logging.info(f"✅ Sheet written: {sheet}")
                else:
                    logging.warning(f"⚠️ Skipped non-DataFrame result: {sheet}")

        logging.info("🎉 Analysis and export completed successfully.")

    except Exception as e:
        logging.exception(f"❌ Error during analysis: {e}")
        print("❌ Analysis failed. Check log for details.")

if __name__ == "__main__":
    run_analysis()
