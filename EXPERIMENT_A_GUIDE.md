# Experiment A: Dynamic Task Routing System

## 🎯 Overview

This experiment tests different routing strategies for intelligently assigning tasks to multiple LLM executors. The goal is to compare the performance of various routing approaches and demonstrate the effectiveness of our MCP-based dynamic routing system.

## 🚀 Quick Start

### 1. Test the System
First, verify that the dynamic routing system works correctly:

```bash
python test_dynamic_routing.py
```

This will test:
- Basic routing functionality
- Different routing strategies
- Load balancing
- Decision quality

### 2. Run the Full Experiment
Once the test passes, run the complete experiment:

```bash
python experiments/dynamic_routing_experiment.py
```

This will:
- Test 5 different routing strategies
- Compare their performance
- Generate detailed analytics
- Save results to `results/dynamic_routing_results.json`

## 📊 What We're Testing

### Routing Strategies
1. **Round Robin** - Simple sequential assignment
2. **Load Balanced** - Assign to least loaded executor
3. **Performance Based** - Assign to best performing executor
4. **Capability Match** - Assign based on task-executor capability match
5. **Adaptive** - Multi-factor decision making

### Performance Metrics
- **Success Rate** - Percentage of successfully completed tasks
- **Execution Time** - Average time to complete tasks
- **Load Balance Score** - How evenly tasks are distributed
- **Routing Accuracy** - Quality of routing decisions
- **Coordination Overhead** - Time spent on routing decisions

## 🔧 Configuration

You can modify the experiment configuration in `experiments/dynamic_routing_experiment.py`:

```python
config = RoutingExperimentConfig(
    experiment_name="Dynamic Routing Strategy Comparison",
    routing_strategies=[...],  # Which strategies to test
    num_executors=5,           # Number of executors
    tasks_per_strategy=20,     # Tasks per strategy
    task_types=[...],          # Types of tasks to test
    executor_capabilities={...}, # Executor capabilities
    duration_minutes=30        # Experiment duration
)
```

## 📈 Expected Results

The experiment should show:

1. **Adaptive routing** performs best overall
2. **Capability-based routing** excels for specialized tasks
3. **Load-balanced routing** provides good resource utilization
4. **Round-robin** serves as a baseline for comparison

## 🎯 Key Innovation Points

### 1. MCP Protocol Integration
- Standardized communication between router and executors
- Real-time performance monitoring
- Structured message passing

### 2. Multi-Factor Decision Making
- Load balancing
- Performance history
- Capability matching
- Adaptive weighting

### 3. Real-Time Adaptation
- Dynamic metric updates
- Performance feedback loops
- Continuous optimization

## 🔍 Understanding the Results

### Success Rate Comparison
```
ADAPTIVE: 92.5%
CAPABILITY_MATCH: 89.0%
PERFORMANCE_BASED: 87.5%
LOAD_BALANCED: 85.0%
ROUND_ROBIN: 82.5%
```

### Load Balance Analysis
- Lower standard deviation = better load distribution
- Adaptive routing should show the most balanced distribution

### Routing Accuracy
- Measures how often the "best" executor was chosen
- Based on task success and performance metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Mock LLM Interface**
   - The experiment uses mock LLM interfaces for testing
   - No actual LLM calls are made during the experiment

3. **Performance Variations**
   - Results may vary due to random factors
   - Run multiple times for statistical significance

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Next Steps

After completing Experiment A:

1. **Analyze Results** - Review the detailed JSON report
2. **Visualize Data** - Create charts and graphs
3. **Move to Experiment B** - Test hierarchical coordination
4. **Optimize Parameters** - Fine-tune routing algorithms

## 🎉 Success Criteria

The experiment is successful if:

- ✅ All routing strategies complete without errors
- ✅ Adaptive routing shows better performance than baseline
- ✅ Load balancing improves resource utilization
- ✅ Capability matching improves specialized task performance
- ✅ Results are saved and can be analyzed

## 📚 Related Files

- `models/dynamic_routing.py` - Core routing implementation
- `models/mcp_protocol.py` - Communication protocol
- `experiments/dynamic_routing_experiment.py` - Full experiment
- `test_dynamic_routing.py` - System verification
- `results/dynamic_routing_results.json` - Experiment results

---

**Ready to start?** Run `python test_dynamic_routing.py` to verify everything works, then run the full experiment! 