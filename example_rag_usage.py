# example_rag_usage.py
from rag import ask

if __name__ == "__main__":
    question = "Примеры мест зон отдыха и развлечений?"
 
    answer = ask(question)
    print("Вопрос:", question)
    print("Ответ:", answer)