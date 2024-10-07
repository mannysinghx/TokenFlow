# TokenFlow
Overview

Tokenflow is a modular, flexible, and robust Python library inspired by AdalFlow and PyTorch. It is designed to simplify the creation and optimization of Large Language Model (LLM) task pipelines. The library provides model-agnostic building blocks that can be used to create and optimize tasks for LLMs, ranging from Retrieval-Augmented Generation (RAG) and agents to classical NLP tasks like text classification and named entity recognition.

The library's design embraces the flexibility of PyTorch, making it token-efficient and easy to work with, enabling users to optimize prompts, manage training workflows, and perform zero-shot or few-shot learning with ease.

Features

Model-Agnostic Pipeline Components: Provides base classes like FlowModule, FlowTrainer, and FlowDataset to build LLM task pipelines with minimal abstraction and maximum flexibility.

Unified Framework for Optimization: Supports both zero-shot prompt optimization and few-shot optimization, providing a unified interface for diagnostics, visualization, debugging, and training.

Token Efficiency: Leverages efficient token utilization for training and inference, ensuring that models achieve high performance without wasting computational resources.

Automatic Graph Tracing: Automatically traces the computation graph, saving developers from having to manually define nodes and edges.

Prompt Optimization: Built-in support for prompt optimization, enabling users to identify the most effective prompts for their LLMs through trials.

Parameter Management: Easy parameter management through the Generator class, allowing for customizable optimization processes.

Installation

To install the AdalFlow Clone library, use the following command:

Usage

Example Usage

Below is a basic usage example that demonstrates how to train and optimize an LLM pipeline using the AdalFlow Clone library.

Key Classes

FlowModule: A base class that provides the building block for model components.

Handles the tokenization, forward pass, and automatic computation graph tracing.

FlowTrainer: Responsible for training the FlowModule with zero-shot and few-shot learning capabilities.

Features gradient clipping, learning rate scheduling, and error handling.

FlowDataset: A modular dataset that supports different types of LLM tasks.

FlowOptimizer: Manages optimization across multiple trials, finding the best model configuration.

PromptTuner: Provides tools for prompt optimization through multiple trials.

Generator: Manages parameterized optimization to further improve the performance of the pipeline.

Advanced Features

Token-Efficient Prompt Optimization

AdalFlow Clone provides built-in support for optimizing prompts. You can utilize PromptTuner to evaluate and improve prompts to get the most effective response from your LLM.

Few-Shot Learning Support

The FlowTrainer class has built-in support for few-shot learning, making it easy to train models on a small number of labeled examples and improve accuracy efficiently.

Parameter Optimization

Using the Generator class, you can easily define and optimize pipeline parameters like learning rates and batch sizes for better performance.

Installation from Source

To install the library from source:

Clone the repository:

Navigate into the cloned directory:

Install the package:

Requirements

Python >= 3.7

PyTorch

Transformers (HuggingFace)

Contributing

We welcome contributions from the community! Please feel free to submit issues, fork the repository, and open pull requests.

License

This project is licensed under the MIT License.

Acknowledgements

The library is inspired by the AdalFlow framework and the PyTorch ecosystem.

The HuggingFace Transformers library is used for model tokenization and pre-trained models.

Contact

For any questions, feel free to reach out to the project maintainers at manindersinghx@gmail.com
