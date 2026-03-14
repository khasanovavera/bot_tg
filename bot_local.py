import os

import telebot
from telebot import apihelper
from dotenv import load_dotenv

from rag import ask


load_dotenv()

# Прокси для доступа к api.telegram.org
# (если без прокси/VPN бот не подключается)
# В .env добавьте: TG_PROXY=http://127.0.0.1:ПОРТ или socks5://127.0.0.1:ПОРТ
proxy = os.getenv("TG_PROXY")
if proxy:
    apihelper.proxy = {"https": proxy, "http": proxy}

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(func=lambda message: True)
def handle_rag_message(message):
    try:
        user_text = (message.text or "").strip()
        if not user_text:
            bot.reply_to(message, "Напишите текст сообщения.")
            return

        print(user_text)
        response = ask(user_text)
        print(response)

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


if __name__ == "__main__":
    bot.polling()
