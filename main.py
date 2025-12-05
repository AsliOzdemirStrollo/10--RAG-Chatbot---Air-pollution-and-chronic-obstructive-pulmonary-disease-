import logging

# --- Silence noisy library logs ---
logging.getLogger().setLevel(logging.WARNING)        # Root logger
logging.getLogger("llama_index").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("generativelanguage").setLevel(logging.ERROR)

from src.engine import main_chat_loop


def main() -> None:
    print("--- 🤖 Main Application Starting ---")
    main_chat_loop()


if __name__ == "__main__":
    main()