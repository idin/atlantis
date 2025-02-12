import pytest
import os
from langchain_openai import ChatOpenAI
from langchain_community.memory.kg import ConversationKGMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from atlantis.ai.KnowledgeGraphMemory import KnowledgeGraphMemory



from langchain.memory import ConversationSummaryMemory

@pytest.fixture
def conversation_v7():
    """Initialize the seventh version using ConversationSummaryMemory for explicit recall."""

    # Ensure OpenAI API key is correctly accessed
    assert "OPENAI_API_KEY" in os.environ, "OpenAI API key is missing from environment variables!"

    # Initialize the chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

    # ✅ Use ConversationSummaryMemory instead of ConversationKGMemory
    memory = ConversationSummaryMemory(llm=llm, return_messages=True)

    return {"llm": llm, "memory": memory}

def test_knowledge_graph_memory_v7(conversation_v7):
    """Test if the seventh version correctly stores and retrieves memory."""

    llm = conversation_v7["llm"]
    memory = conversation_v7["memory"]

    # Step 1: Store Context in Memory
    memory.save_context(
        {"input": "My name is Idin and I love playing bass."},
        {"output": "Got it!"}
    )

    # Step 2: Retrieve Stored Memory and Print
    stored_memory = memory.load_memory_variables({})
    print("DEBUG: Stored Memory After Save Context:", stored_memory)

    # Ensure memory contains key details
    assert "Idin" in str(stored_memory), "The memory should contain the name 'Idin'"
    assert "bass" in str(stored_memory), "The memory should contain 'playing bass'"

    # Step 3: Retrieve History Explicitly
    retrieved_history = stored_memory.get("history", "")
    print("DEBUG: Retrieved History Before Querying Model:", retrieved_history)

    # Step 4: Query Model with Retrieved Memory
    full_prompt = f"{retrieved_history}\nWhat is my name and what do I love?"
    response2 = llm.invoke(full_prompt)

    print("DEBUG: Model Response:", response2)

    # Ensure the model recalls the stored details
    assert "Idin" in response2.content, "The model should recall the name 'Idin'"
    assert "bass" in response2.content, "The model should recall 'playing bass'"


@pytest.fixture
def conversation_v8():
    """Initialize a working memory system using a simple Python dictionary."""

    # Ensure OpenAI API key is correctly accessed
    assert "OPENAI_API_KEY" in os.environ, "OpenAI API key is missing from environment variables!"

    # Initialize the chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

    # ✅ Use a simple dictionary as memory
    memory_store = {"history": ""}

    def save_to_memory(user_input, response):
        """Manually save context to memory."""
        memory_store["history"] += f"\nUser: {user_input}\nAI: {response}"

    def get_memory():
        """Retrieve stored history."""
        return memory_store["history"]

    return {"llm": llm, "save_to_memory": save_to_memory, "get_memory": get_memory}

def test_knowledge_graph_memory_v8(conversation_v8):
    """Test if the manually implemented memory system actually works."""

    llm = conversation_v8["llm"]
    save_to_memory = conversation_v8["save_to_memory"]
    get_memory = conversation_v8["get_memory"]

    # Step 1: Store Context in Memory Manually
    user_input = "My name is Idin and I love playing bass."
    response1 = "Got it!"
    save_to_memory(user_input, response1)

    # Step 2: Retrieve Stored Memory
    stored_memory = get_memory()
    print("DEBUG: Stored Memory After Save Context:", stored_memory)

    # Ensure memory contains key details
    assert "Idin" in stored_memory, "The memory should contain the name 'Idin'"
    assert "bass" in stored_memory, "The memory should contain 'playing bass'"

    # Step 3: Retrieve History and Inject It Before Querying the Model
    retrieved_history = get_memory()
    full_prompt = f"{retrieved_history}\nWhat is my name and what do I love?"
    
    # Step 4: Query Model
    response2 = llm.invoke(full_prompt)
    print("DEBUG: Model Response:", response2)

    # Ensure the model recalls the stored details
    assert "Idin" in response2.content, "The model should recall the name 'Idin'"
    assert "bass" in response2.content, "The model should recall 'playing bass'"


def test_debug_model_response(conversation_v8):
    """Check what OpenAI is actually returning instead of assuming memory recall works."""

    llm = conversation_v8["llm"]
    save_to_memory = conversation_v8["save_to_memory"]
    get_memory = conversation_v8["get_memory"]

    # Step 1: Store Context in Memory Manually
    user_input = "My name is Idin and I love playing bass."
    response1 = "Got it!"
    save_to_memory(user_input, response1)

    # Step 2: Retrieve Stored Memory
    stored_memory = get_memory()
    print("DEBUG: Stored Memory After Save Context:", stored_memory)

    # Step 3: Retrieve History and Inject It Before Querying the Model
    retrieved_history = get_memory()
    full_prompt = f"{retrieved_history}\nWhat is my name and what do I love?"
    
    # Step 4: Query Model
    print("DEBUG: Full Prompt Sent to OpenAI:", full_prompt)
    response2 = llm.invoke(full_prompt)

    print("DEBUG: Model Response:", response2.content)  # FIXED: Print only the message content

    # Step 5: Assert model response actually contains expected memory
    assert "Idin" in response2.content, "ERROR: The model did NOT recall 'Idin'."  # FIXED
    assert "bass" in response2.content, "ERROR: The model did NOT recall 'playing bass'."  # FIXED


@pytest.fixture
def conversation_v9():
    """Initialize KnowledgeGraphMemory with an LLM."""

    assert "OPENAI_API_KEY" in os.environ, "OpenAI API key is missing from environment variables!"

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"])

    memory = KnowledgeGraphMemory(llm=llm)  # Pass LLM when initializing

    return {"llm": llm, "memory": memory}


def test_knowledge_graph_memory_v9(conversation_v9):
    """Test if KnowledgeGraphMemory correctly stores and retrieves memory."""

    llm = conversation_v9["llm"]
    memory = conversation_v9["memory"]

    # Step 1: Store Context in Knowledge Graph Memory
    memory.save_context(
        {"input": "Idin loves playing bass."},
        {"output": "Got it!"}
    )

    # Step 2: Retrieve Stored Memory and Print
    stored_memory = memory.load_memory_variables({"input": "Idin"})
    print("DEBUG: Stored Memory After Save Context:", stored_memory)

    # Ensure memory contains key details
    assert "Idin loves playing bass" in stored_memory["history"], "The memory should contain the fact about Idin."

    # Step 3: Query Model with Retrieved Memory
    retrieved_history = stored_memory["history"]
    full_prompt = f"{retrieved_history}\nWhat does Idin love?"
    response2 = llm.invoke(full_prompt)

    print("DEBUG: Model Response:", response2.content)

    # Ensure the model recalls the stored details
    assert "bass" in response2.content, "The model should recall 'playing bass'."
