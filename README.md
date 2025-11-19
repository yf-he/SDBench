# SDBench - Sequential Diagnosis Benchmark

SDBench is a comprehensive benchmark system for evaluating diagnostic reasoning capabilities of AI agents and humans in a realistic, iterative, and cost-aware manner. This implementation reproduces the framework described in the paper "Sequential Diagnosis with Language Models" by Harsha Nori*, Mayank Daswani*, Christopher Kelly*, Scott Lundberg*, Marco Tulio Ribeiro*, Marc Wilson*, Xiaoxuan Liu, Viknesh Sounderajah, Jonathan M Carlson, Matthew P Lungren, Bay Gross, Peter Hames, Mustafa Suleyman, Dominic King, Eric Horvitz (Microsoft AI, July 3, 2025) and provides a multi-agent system that simulates clinical encounters.

## Overview

SDBench consists of three core components that interact with a diagnostic agent:

1. **Gatekeeper Agent** - Serves as the information oracle for patient cases
2. **Cost Estimator** - Calculates monetary costs of diagnostic processes
3. **Judge Agent** - Evaluates final diagnoses using a 5-point Likert scale

## Features

- **Sequential Interaction**: Turn-based diagnostic encounters with realistic clinical flow
- **Cost Awareness**: Tracks both physician visit costs and test costs using CPT codes
- **Synthetic Data Generation**: Generates plausible clinical findings when information isn't explicitly available
- **Comprehensive Evaluation**: Uses a 5-point rubric for diagnostic accuracy assessment
- **Multiple Agent Types**: Includes example agents for testing and comparison
- **Interactive UI Demos**: Multiple Streamlit interfaces including the Multi-turn Medical Diagnosis Copilot for human-AI collaborative diagnosis

## Installation

1. Clone or download the SDBench repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For the interactive UI demos, also install Streamlit:
   ```bash
   pip install streamlit
   ```
3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   
   For OpenRouter support (alternative to OpenAI), set:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   export SDBENCH_API_PROVIDER="openrouter"
   ```

## Quick Start

### Run a Quick Test
```bash
python main.py quick
```

### Run Single Case Demo
```bash
python main.py single
```

### Run Full Benchmark
```bash
python main.py full
```

### Run Interactive Demo
```bash
python main.py interactive
```

## Interactive UI Demos

SDBench includes several Streamlit-based interactive interfaces for exploring the diagnostic system:

### Multi-turn Medical Diagnosis Copilot (`clinical_rounds_ui.py`)

A comprehensive interactive demo that simulates a real-world clinical diagnostic encounter with human-AI collaboration.

**Features:**
- **Role-playing scenario**: You play the role of an attending physician making diagnostic decisions
- **AI Resident Doctor**: An AI assistant that can draft questions, test orders, or diagnoses on demand
- **Medical Evidence System**: Provides patient information, answers clinical questions, and returns test results
- **Flexible workflow**: Choose to draft each step yourself or let the AI resident doctor suggest the next action
- **Real-time evaluation**: Receive expert evaluation of your final diagnosis against ground truth

**Usage:**
```bash
streamlit run clinical_rounds_ui.py
```

**How it works:**
1. Select a patient case from the dataset
2. Review the initial presentation (chief complaint and findings)
3. For each turn, choose to:
   - Ask questions about patient history, symptoms, or examination findings
   - Order diagnostic tests or imaging studies
   - Submit a final diagnosis
4. Decide whether to draft the action yourself or let the AI resident doctor suggest a draft
5. Receive responses from the Medical Evidence System
6. Finalize your diagnosis to receive expert evaluation

This creates a collaborative human-AI workflow that mirrors real clinical decision-making processes.

### Human Agent UI (`human_agent_ui.py`)

A simpler interface for manual diagnostic interaction without AI assistance.

**Usage:**
```bash
streamlit run human_agent_ui.py
```

### LLM Agent UI (`llm_agent_ui.py`)

An interface for running and visualizing fully automated LLM diagnostic agents.

**Usage:**
```bash
streamlit run llm_agent_ui.py
```

## System Architecture

### Core Components

#### Gatekeeper Agent
- Uses GPT-4o-mini for information retrieval and synthesis
- Provides explicit information when available in case files
- Generates synthetic findings when information is missing
- Enforces strict gatekeeping rules to prevent information leakage

#### Cost Estimator
- Maps test requests to CPT codes using language models
- Uses 2023 CMS pricing data for cost calculation
- Tracks both physician visit costs ($300 per visit) and test costs
- Provides fallback estimation for unmapped procedures

#### Judge Agent
- Uses GPT-4o for diagnosis evaluation
- Implements the exact 5-point Likert scale from the paper:
  - 5: Perfect/Clinically superior
  - 4: Mostly correct (minor incompleteness)
  - 3: Partially correct (major error)
  - 2: Largely incorrect
  - 1: Completely incorrect

### Data Models

The system uses Pydantic models for type safety and validation:
- `CaseFile`: Complete case information including abstract and full text
- `AgentAction`: Actions taken by diagnostic agents
- `DiagnosticEncounter`: Complete encounter with all interactions
- `BenchmarkResult`: Evaluation results and metrics

## Example Agents

The system includes several example diagnostic agents:

1. **RandomDiagnosticAgent**: Takes random actions for baseline testing
2. **ConservativeDiagnosticAgent**: Asks many questions before ordering tests
3. **AggressiveDiagnosticAgent**: Orders many tests quickly
4. **LLMDiagnosticAgent**: Uses language models for intelligent decision-making

## Synthetic Cases

The system includes three synthetic NEJM-style cases for testing:

1. **Histoplasmosis with mediastinal lymphadenopathy** - A 34-year-old woman with respiratory symptoms
2. **Autoimmune hemolytic anemia** - A 28-year-old woman with fatigue and jaundice
3. **Pheochromocytoma** - A 45-year-old man with episodic hypertension

## Usage Examples

### Basic Benchmark Run
```python
from config import Config
from sdbench import SDBench
from example_agents import RandomDiagnosticAgent
from synthetic_cases import get_all_synthetic_cases

# Setup
config = Config()
sdbench = SDBench(config)

# Create agent and cases
agent = RandomDiagnosticAgent("MyAgent")
cases = get_all_synthetic_cases()

# Run benchmark
result = sdbench.run_benchmark(agent, cases)
print(f"Accuracy: {result.diagnostic_accuracy:.2%}")
print(f"Average Cost: ${result.average_cost:.2f}")
```

### Custom Diagnostic Agent
```python
from sdbench import DiagnosticAgent
from data_models import AgentAction, ActionType

class MyCustomAgent(DiagnosticAgent):
    def __init__(self):
        super().__init__("MyCustomAgent")
    
    def get_next_action(self, case_abstract: str, encounter_history: List[AgentAction]) -> AgentAction:
        # Implement your diagnostic logic here
        if len(encounter_history) < 5:
            return AgentAction(
                action_type=ActionType.ASK_QUESTIONS,
                content="What are the patient's vital signs?"
            )
        else:
            return AgentAction(
                action_type=ActionType.DIAGNOSE,
                content="Based on available information, the diagnosis is..."
            )
    
    def reset(self):
        # Reset agent state for new case
        pass
```

## Evaluation Metrics

### Primary Metric: Diagnostic Accuracy
- Percentage of cases with judge score ≥ 4
- `Accuracy = (Correct Cases) / (Total Cases)`

### Secondary Metric: Average Cumulative Cost
- Average total cost across all cases
- Includes physician visits and test costs

### Performance Visualization
The system generates 2D plots showing the trade-off between cost and accuracy, enabling Pareto frontier analysis.

## Configuration

Key configuration options in `config.py`:

- `PHYSICIAN_VISIT_COST`: Cost per physician visit (default: $300)
- `CORRECT_DIAGNOSIS_THRESHOLD`: Minimum score for correct diagnosis (default: 4)
- `GATEKEEPER_MODEL`: Model for gatekeeper agent (default: "gpt-4o-mini")
- `JUDGE_MODEL`: Model for judge agent (default: "gpt-4o")

## File Structure

```
SDBench/
├── config.py              # Configuration settings
├── data_models.py         # Pydantic data models
├── gatekeeper_agent.py    # Gatekeeper agent implementation
├── cost_estimator.py      # Cost estimation module
├── judge_agent.py         # Judge agent implementation
├── evaluation_protocol.py # Evaluation metrics and reporting
├── sdbench.py            # Main SDBench class
├── synthetic_cases.py    # Synthetic test cases
├── example_agents.py     # Example diagnostic agents
├── main.py               # Main script and demos
├── clinical_rounds_ui.py # Multi-turn Medical Diagnosis Copilot UI
├── human_agent_ui.py     # Human agent interactive UI
├── llm_agent_ui.py       # LLM agent visualization UI
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Contributing

To contribute to SDBench:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project reproduces the SDBench framework as described in "Sequential Diagnosis with Language Models" by Harsha Nori, Mayank Daswani, Christopher Kelly, Scott Lundberg, Marco Tulio Ribeiro, Marc Wilson, Xiaoxuan Liu, Viknesh Sounderajah, Jonathan M Carlson, Matthew P Lungren, Bay Gross, Peter Hames, Mustafa Suleyman, Dominic King, and Eric Horvitz (Microsoft AI, July 2025) and is intended for research and educational purposes.

## Citation

If you use SDBench in your research, please cite the original paper:

```
@article{nori2025sequential,
  title={Sequential Diagnosis with Language Models},
  author={Harsha Nori and Mayank Daswani and Christopher Kelly and Scott Lundberg and Marco Tulio Ribeiro and Marc Wilson and Xiaoxuan Liu and Viknesh Sounderajah and Jonathan M Carlson and Matthew P Lungren and Bay Gross and Peter Hames and Mustafa Suleyman and Dominic King and Eric Horvitz},
  journal={Microsoft AI},
  year={2025},
  month={July}
}
```

## Support

For questions or issues, please open an issue on the repository or contact the development team.
