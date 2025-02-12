import openai
import time
from langchain.memory import ConversationKGMemory, VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda  # ✅ Correct fix
from typing import Union, List

class ChatGPT:
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.5, semantic_search: bool = True, knowledge_graph: bool = True):
        self.start_time = time.time()
        self._api_key = api_key
        self.model = model
        self.temperature = temperature

        # ✅ Use `RunnableLambda` to properly wrap OpenAI API calls
        self.llm = RunnableLambda(self._call_openai)
        
        self.main_memory = Memory(self, semantic_search=semantic_search, knowledge_graph=knowledge_graph)

    def _call_openai(self, prompt: str) -> str:
        """Calls OpenAI API and returns the response."""
        response = openai.ChatCompletion.create(
            api_key=self._api_key,
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]

    def conversation(self, memory: Union[bool, "Memory", None, str] = 'new', semantic_search: bool = True, knowledge_graph: bool = True):
        """Creates a conversation object with its own memory."""
        if memory is False:
            return Conversation(ai=self, memory=None)
        elif memory is True or memory == 'new':
            return Conversation(ai=self, memory=Memory(self, semantic_search=semantic_search, knowledge_graph=knowledge_graph))
        elif memory is None or memory == 'ai' or memory == 'self' or memory is self:
            return self.main_conversation
        elif isinstance(memory, Memory):
            return Conversation(ai=self, memory=memory)
        else:
            raise TypeError(f"Invalid memory argument: {memory}, of type {type(memory)}")

    def prompt(self, prompt: str) -> str:
        return self.main_memory.prompt(prompt)

    def prompt_without_memory(self, prompt: str) -> str:
        """Send a direct request to OpenAI without memory."""
        return self._call_openai(prompt)

class Memory:
    """A class that initializes and manages different types of memory automatically."""

    @property
    def llm(self):
        return self.ai.llm  # ✅ Now correctly returns a LangChain-compatible LLM

    def __init__(self, ai: ChatGPT, semantic_search: bool = True, knowledge_graph: bool = True):
        self.ai = ai
        self.knowledge_graph = None
        self.semantic_search = None
        self.retriever = None

        # 🔹 Initialize Knowledge Graph Memory (Stores structured knowledge)
        if knowledge_graph:
            self.knowledge_graph = ConversationKGMemory(llm=self.llm)  # ✅ Now `llm` is correctly formatted

        # 🔹 Initialize Vector Store Memory (Stores past conversations)
        if semantic_search:
            embedding_function = OpenAIEmbeddings(api_key=self.ai._api_key)
            vector_store = FAISS.from_texts(texts=[""], embedding=embedding_function)
            self.retriever = vector_store.as_retriever()
            self.semantic_search = VectorStoreRetrieverMemory(retriever=self.retriever)

        # 🔹 Automatically combine both memories using ConversationalRetrievalChain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever if self.semantic_search else None,
            memory=self.knowledge_graph if self.knowledge_graph else None
        )

    def prompt(self, query: str) -> str:
        """Send a query to AI and let LangChain decide which memory to use."""
        if self.semantic_search or self.knowledge_graph:
            return self.chain.invoke({"question": query})
        else:
            return self.llm.invoke(query)

    def prompt_without_memory(self, query: str) -> str:
        """Send a direct request to OpenAI without using memory."""
        return self.llm.invoke(query)

class Conversation:
    """A class that contains its own memory and can be used to have a conversation with the AI."""

    def __init__(self, ai: ChatGPT, memory: Memory):
        self.start_time = time.time()
        self.ai = ai
        self.memory = memory

    def prompt(self, prompt: str) -> str:
        return self.memory.prompt(prompt)

    def prompt_without_memory(self, prompt: str) -> str:
        return self.ai.prompt_without_memory(prompt)
