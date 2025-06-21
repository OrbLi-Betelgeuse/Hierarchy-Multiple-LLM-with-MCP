# Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running locally
3. **Git** for cloning the repository

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Hierarchy-Multiple-LLM-with-MCP
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama models:**
   ```bash
   # Install the models we'll use
   ollama pull llama2:7b
   ollama pull llama2:70b
   ```

## Quick Test

1. **Test the system components:**
   ```bash
   python test_system.py
   ```

2. **Run a simple experiment:**
   ```bash
   python pipeline.py --experiment summarization
   ```

## Running Experiments

### Basic Usage

```bash
# Run all experiments
python pipeline.py

# Run specific experiment
python pipeline.py --experiment summarization
python pipeline.py --experiment qa
python pipeline.py --experiment table

# Use custom configuration
python pipeline.py --config config_example.json

# Specify different models
python pipeline.py --manager-model llama2:70b --executor-model llama2:7b
```

### Configuration

The system uses a JSON configuration file to define:
- Manager LLM settings
- Executor LLM settings
- Experiment parameters
- Output settings

Example configuration:
```json
{
  "manager": {
    "provider": "ollama",
    "model": "llama2:70b",
    "manager_id": "manager_01"
  },
  "executors": [
    {
      "provider": "ollama",
      "model": "llama2:7b",
      "executor_id": "executor_01",
      "capabilities": ["summarization", "general"]
    }
  ],
  "experiments": {
    "summarization": {"enabled": true, "num_tasks": 2},
    "question_answering": {"enabled": true, "num_tasks": 2},
    "table_generation": {"enabled": true, "num_tasks": 2}
  }
}
```

## Experiment Types

### 1. Long Document Summarization
- **Purpose**: Test the system's ability to break down and summarize long documents
- **Input**: Long text documents
- **Output**: Concise summaries with quality metrics
- **Metrics**: Compression ratio, ROUGE scores, execution time

### 2. Multi-round Question Answering
- **Purpose**: Test conversational reasoning capabilities
- **Input**: Context + multiple questions
- **Output**: Sequential answers with context awareness
- **Metrics**: Accuracy, response coherence, execution time

### 3. Structured Table Generation
- **Purpose**: Test structured data extraction and formatting
- **Input**: Unstructured text with data
- **Output**: Formatted tables (Markdown, CSV, JSON)
- **Metrics**: Structure accuracy, format compliance, execution time

## Understanding Results

After running experiments, results are saved in the `results/` directory:

- `comprehensive_report.json`: Overall system performance
- `summarization_metrics.json`: Summarization-specific metrics
- `question_answering_metrics.json`: QA-specific metrics
- `table_generation_metrics.json`: Table generation metrics

### Key Metrics

- **Success Rate**: Percentage of tasks completed successfully
- **Execution Time**: Average time per task
- **Quality Score**: Task-specific quality assessment
- **Resource Utilization**: CPU, memory, token usage

## Troubleshooting

### Common Issues

1. **Ollama not running:**
   ```bash
   # Start Ollama
   ollama serve
   ```

2. **Model not found:**
   ```bash
   # Pull required models
   ollama pull llama2:7b
   ollama pull llama2:70b
   ```

3. **Import errors:**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

4. **Memory issues:**
   - Use smaller models (llama2:7b instead of llama2:70b)
   - Reduce number of concurrent executors
   - Increase system memory

### Getting Help

- Check the logs in `experiment.log`
- Run `python test_system.py` to verify setup
- Review the configuration file format

## Next Steps

1. **Customize experiments** by modifying the configuration
2. **Add new task types** by extending the experiment modules
3. **Integrate with RAGFlow** for enhanced document processing
4. **Scale the system** by adding more executors or using cloud LLMs

## Architecture Overview

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

The system follows a hierarchical Manager-Executor pattern where:
- **Manager**: High-capacity LLM responsible for task decomposition and coordination
- **Executors**: Lightweight LLMs focused on specific subtasks
- **MCP**: Model Context Protocol for structured communication 