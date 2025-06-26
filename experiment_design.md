## 一、实验设计思路

  
### 1. 目标

- 构建一个**分层多LLM系统**，通过Manager-Executor协作模式提升自然语言任务的效率和效果。

- 集成**RAGFlow**，实现检索增强生成（Retrieval-Augmented Generation, RAG），让LLM不仅能生成，还能基于知识库检索相关文档，提升答案的准确性和可溯源性。

  

### 2. 实验类型
- **长文档摘要**：Manager将长文档分解为可管理的子任务，Executor负责具体的摘要生成。
- **多轮问答**：Manager负责多轮对话的上下文管理和任务分配，Executor负责具体问答。
- **结构化表格生成**：Manager分解表格生成任务，Executor负责信息抽取与表格填充。
- **RAG实验**（新加）：通过RAGFlow，先检索相关文档，再由LLM基于检索结果生成答案，适用于问答、摘要、研究类任务。

  

### 3. RAG实验流程
1. **知识库构建**：上传一批领域文档到RAGFlow，构建知识库。
2. **任务定义**：设计一组需要知识检索支撑的问答/摘要/研究任务。
3. **检索与生成**：RAGExecutor先用RAGFlow检索相关文档，再将检索结果与原始query一起送入LLM生成最终答案。
4. **评估**：对比RAG与非RAG的效果，评估准确率、召回率、生成质量等。

  

  

## 二、代码实现思路

  

### 1. 系统结构

- **Manager**：负责任务分解、规划、分配，维护Executor注册表。

- **Executor**：负责具体子任务的执行，支持普通LLM和RAG两种模式。

- **RAGExecutor**：继承自Executor，集成RAGFlow接口，支持检索增强生成。

- **RAGFlowInterface**：封装RAGFlow的API，包括知识库管理、文档上传、检索、生成等。

- **Experiment模块**：每种实验类型一个模块，负责实验流程、数据准备、结果评估。

  

### 2. RAGFlow集成关键点

- **RAGFlowInterface**：实现知识库创建、文档上传、检索、生成等方法，所有操作均为异步，便于与主流程集成。

- **RAGExecutor**：

  - 初始化时注入RAGFlowInterface和知识库ID。

  - 执行任务时，先用RAGFlow检索相关文档，再将检索结果与query拼接，送入LLM生成答案。

  - 支持fallback：如果RAG不可用或检索不到文档，自动降级为普通LLM生成。

- **RAGExperiment**：

  - 实验setup时自动创建知识库并上传样本文档。

  - 每个任务先用RAGExecutor执行，记录检索文档、置信度、生成内容等。

  - 支持质量评估（如内容重叠率、召回率等）。

  

### 3. 代码流程举例（以RAG问答为例）

1. **初始化RAGExecutor**，并setup知识库、上传文档。

2. **定义任务**：如“什么是人工智能？”。

3. **执行任务**：

   - RAGExecutor调用RAGFlow检索相关文档。

   - 将检索到的文档内容与问题拼接，作为prompt输入LLM。

   - LLM生成答案，返回结果、检索文档、置信度等。

4. **评估与报告**：统计每个任务的执行时间、置信度、召回文档数、答案质量等，生成实验报告。


### 4. 代码可扩展性
- 支持多种LLM（Ollama、OpenAI等），只需实现对应的LLMInterface。
- RAGFlow接口解耦，未来可替换为其他RAG系统。
- Manager-Executor模式可扩展为多层级、多Agent协作。
