# Manager-Executor Collaboration Model

A hierarchical multi-LLM collaboration system that simulates organizational structures to enhance natural language task efficiency through task decomposition and role specialization.

## Overview

This project implements a "Manager-Executor" collaboration model where:
- **Manager LLMs**: High-capacity models responsible for task decomposition, planning, and oversight
- **Executor LLMs**: Lightweight models focused on completing specific subtasks
- **MCP Integration**: Model Context Protocol for structured communication between agents

## Tasks Evaluated

1. **Long Document Summarization**: Breaking down large documents into manageable chunks for efficient processing
2. **Multi-round Question Answering**: Handling complex conversational queries through structured reasoning
3. **Structured Table Generation**: Creating formatted data outputs from unstructured text

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Manager LLM   │────│   MCP Protocol  │────│  Executor LLMs  │
│   (Planning)    │    │  (Communication)│    │   (Execution)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Task Decomposition    Structured Messages    Subtask Execution
   Role Assignment       Coordination           Result Aggregation
```

## Setup

### Prerequisites
- Python 3.8+
- Ollama (for local LLM inference)
- Docker (for RAGFlow)

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
python pipeline.py --task summarization --model-manager llama2:70b --model-executor llama2:7b
```

## Project Structure
- `pipeline.py`: Main experimental pipeline
- `experiments/`: Task-specific experiment modules
- `models/`: LLM interaction and MCP implementation
- `utils/`: Utility functions and evaluation metrics
- `data/`: Test datasets and examples
