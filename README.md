# Atlantis

Atlantis is a powerful framework designed for building AI applications with advanced memory management capabilities. It provides a flexible architecture for integrating various AI models, memory systems, and data management solutions.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
  - [AI Module](#ai-module)
  - [Memory Module](#memory-module)
- [Contributing](#contributing)
- [License](#license)

## Overview

Atlantis enables developers to create intelligent applications that can remember context, manage knowledge graphs, and interact with various AI models. The framework is designed to be modular, allowing for easy integration of new components and functionalities.

## Installation

To install Atlantis, you can use pip:

```bash
pip install atlantis
```

Make sure to have the required dependencies installed. You can find the list of dependencies in the `setup.py` file.

## Usage

Here's a simple example of how to use Atlantis to create a ChatGPT instance and interact with it:

```python
from atlantis.ai_interface.ChatGPT import ChatGPT

# Initialize the ChatGPT instance with your API key
chatgpt = ChatGPT(api_key="your_api_key_here")

# Send a prompt to the model
response = chatgpt.prompt("Hello, how are you?")
print(response)
```

## Modules

### AI Module

The AI module provides interfaces and implementations for various AI models, including support for OpenAI's GPT models. It allows for easy integration and interaction with these models.

### Memory Module

The memory module provides a robust implementation of knowledge graphs and memory management systems. It includes classes and methods for storing, retrieving, and managing facts in a structured manner.

- **BaseKnowledgeGraphDB**: A class for creating a knowledge graph database that supports fact storage and retrieval.
- **Fact**: Represents a single fact in the knowledge graph, encapsulating the subject, predicate, object, and optional metadata.
- **GraphDB**: An interface for interacting with different types of graph databases.

For more detailed information on the memory module, please refer to the [Memory Module Documentation](atlantis/ai/memory/README.md).

## Contributing

We welcome contributions to the Atlantis framework. Please refer to the contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.