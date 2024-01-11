
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

from langchain.memory import ConversationBufferMemory

from langchain_core.runnables import RunnablePassthrough


def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Using the following documents, help answer questions as a teacher would help a student. Remember to only answer the question they asked: {context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# LLM
llm = Ollama(
    model="mistral",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


# question from human
def make_chain(retriever):
    return (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
