import telebot
from telebot import apihelper
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# Прокси для доступа к api.telegram.org (если без прокси/VPN бот не подключается)
# В .env добавьте: TG_PROXY=http://127.0.0.1:ПОРТ или socks5://127.0.0.1:ПОРТ
proxy = os.getenv('TG_PROXY')
if proxy:
    apihelper.proxy = {'https': proxy, 'http': proxy}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"  # LOCAL_API_URL , "https://openrouter.ai/api/v1"
MODEL_NAME = "gemma3:4b"  # "qwen/qwen3-coder:free" openai/gpt-oss-20b:free

llm = ChatOpenAI(openai_api_key='fake_key',
                 openai_api_base=OPENROUTER_BASE,
                 model_name=MODEL_NAME,
                 )
BOT_TOKEN = os.getenv('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    try:
        # Отправляем сообщение в LLM
        print(message.text)
        response = llm.invoke(message.text).content
        print(response)
        # Отвечаем бота ответом LLM
        bot.reply_to(message, response)
    except Exception as e:
        # Обработка ошибок (напр. проблемы с API)
        bot.reply_to(message, f"Ошибка: {str(e)}. Попробуйте позже.")


# Запуск бота
bot.polling()
