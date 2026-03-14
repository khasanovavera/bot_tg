import os

import telebot
from dotenv import load_dotenv

from rag import ask


load_dotenv()

# Замените 'bot_token' на токен вашего бота, сохранённого в .env
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(func=lambda message: True)
def handle_rag_message(message):
    try:
        question = (message.text or "").strip()
        if not question:
            bot.reply_to(message, "зона отдыха и развлечений")

            return

        # Ответ через RAG (knowledge_base.txt + Chroma + Ollama, если доступен)
        response = ask(question)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


if __name__ == "__main__":
    bot.polling()