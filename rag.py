"""
RAG: загрузка базы знаний из knowledge_base.txt, векторный поиск (ChromaDB),
ответы через LLM с контекстом.

Режимы (без .env):
  - С OPENAI_API_KEY: эмбеддинги и GPT от OpenAI.
  - Без ключа: локальные эмбеддинги (sentence-transformers) + Ollama для ответов;
    если Ollama не запущен — возвращаются только найденные фрагменты из базы.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Путь к базе знаний и кэшу векторов
DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = DIR / "knowledge_base.txt"
CHROMA_PERSIST_DIR = DIR / "chroma_db_local"


def _get_embedding():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )


def _get_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2", temperature=0.2)
    except Exception:
        return None  # Ollama не установлен/не запущен — ответ только из ретривера


def load_and_split_knowledge(chunk_size: int = 800, chunk_overlap: int = 150):
    """Загружает knowledge_base.txt и разбивает на чанки."""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]


def get_vectorstore(force_rebuild: bool = False):
    """
    Возвращает векторное хранилище Chroma.
    При первом запуске или force_rebuild=True индексирует knowledge_base.txt.
    """
    from langchain_chroma import Chroma

    embeddings = _get_embedding()
    persist = str(CHROMA_PERSIST_DIR)
    persist_path = CHROMA_PERSIST_DIR
    if force_rebuild and persist_path.exists():
        import shutil
        shutil.rmtree(persist)

    if not persist_path.exists() or force_rebuild:
        docs = load_and_split_knowledge()
        return Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist,
            collection_name="knowledge",
        )

    return Chroma(
        persist_directory=persist,
        embedding_function=embeddings,
        collection_name="knowledge",
    )


def _format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_rag_chain(retriever_k=4):
    """
    Собирает цепочку: вопрос -> поиск релевантных чанков -> ответ LLM по контексту
    (или только релевантные фрагменты, если LLM недоступен).
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    llm = _get_llm()

    if llm is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты помощник, отвечающий только на основе приведённого контекста из базы знаний.
Контекст может быть про Цитадель Риков и связанные темы. Отвечай кратко и по делу.
Если в контексте нет ответа — так и скажи."""),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
        ])
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    # Без LLM: возвращаем только найденные фрагменты (объект с .invoke как у цепочки)
    class RetrievalOnlyChain:
        def invoke(self, question: str) -> str:
            docs = retriever.invoke(question)
            if not docs:
                return "По запросу ничего не найдено в базе знаний."
            return "По вашему запросу найдено в базе знаний:\n\n" + _format_docs(docs)

    return RetrievalOnlyChain()


# Глобальная цепочка (ленивая инициализация)
_chain = None


def ask(question: str, retriever_k: int = 4) -> str:
    """
    Задать вопрос по базе знаний. Использует RAG (поиск + генерация ответа).
    """
    global _chain
    if _chain is None:
        _chain = build_rag_chain(retriever_k=retriever_k)
    return _chain.invoke(question)


def rebuild_index():
    """Пересобрать векторный индекс из knowledge_base.txt."""
    get_vectorstore(force_rebuild=True)


if __name__ == "__main__":
    # Пример использования
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_index()
        print("Индекс пересобран.")
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Где в Цитадели заказать пиццу?"
        print("Вопрос:", q)
        print("Ответ:", ask(q))
