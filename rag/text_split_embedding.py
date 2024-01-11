
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

from langchain_core.runnables import RunnablePassthrough

import vectordb


# Chain
def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)


vectorstore = vectordb.create()
retriever = vectorstore.as_retriever()

# question from human
question = "When was the rock parrot discovered?"
# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Using the following documents, help answer questions as a teacher would help a student. Remember to only answer the question they asked: {context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


# Make sure the model path is correct for your system!
llm = Ollama(
    model="mistral",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

rag_chain = (
    {"context": retriever | format_documents, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke(question)
