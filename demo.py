"""Demo script showing SDBench system structure and capabilities."""

def show_system_overview():
    """Display the SDBench system overview."""
    print("SDBench - Sequential Diagnosis Benchmark")
    print("=" * 50)
    print()
    print("SDBench is a comprehensive benchmark system for evaluating")
    print("diagnostic reasoning capabilities of AI agents and humans")
    print("in a realistic, iterative, and cost-aware manner.")
    print()
    print("Core Components:")
    print("1. Gatekeeper Agent - Information oracle for patient cases")
    print("2. Cost Estimator - Calculates diagnostic process costs")
    print("3. Judge Agent - Evaluates diagnoses using 5-point rubric")
    print("4. Diagnostic Agent - The system being evaluated")
    print()

def show_synthetic_cases():
    """Display information about synthetic cases."""
    print("Synthetic Test Cases")
    print("=" * 20)
    print()
    
    cases = [
        {
            "id": "SYNTH_001",
            "title": "Histoplasmosis with mediastinal lymphadenopathy",
            "abstract": "A 34-year-old woman presents with a 6-week history of progressive dyspnea, dry cough, and fatigue. She reports a 10-pound weight loss over the past month and night sweats. Physical examination reveals bilateral cervical lymphadenopathy and decreased breath sounds at the right lung base.",
            "diagnosis": "Chronic pulmonary histoplasmosis with mediastinal lymphadenopathy"
        },
        {
            "id": "SYNTH_002", 
            "title": "Autoimmune hemolytic anemia",
            "abstract": "A 28-year-old woman presents with a 2-week history of fatigue, jaundice, and dark urine. She reports no recent illness or medication use. Physical examination reveals scleral icterus, pallor, and mild splenomegaly. Laboratory studies show anemia with evidence of hemolysis.",
            "diagnosis": "Autoimmune hemolytic anemia (warm type)"
        },
        {
            "id": "SYNTH_003",
            "title": "Pheochromocytoma",
            "abstract": "A 45-year-old man presents with episodes of severe headaches, palpitations, and diaphoresis lasting 10-15 minutes. The episodes occur 2-3 times per week and are often triggered by stress or physical activity. Physical examination reveals hypertension and tachycardia during an episode.",
            "diagnosis": "Pheochromocytoma of the left adrenal gland"
        }
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"{i}. {case['title']} ({case['id']})")
        print(f"   Abstract: {case['abstract'][:100]}...")
        print(f"   Diagnosis: {case['diagnosis']}")
        print()

def show_agent_types():
    """Display information about example agents."""
    print("Example Diagnostic Agents")
    print("=" * 25)
    print()
    
    agents = [
        {
            "name": "RandomDiagnosticAgent",
            "description": "Takes random actions for baseline testing",
            "strategy": "Randomly chooses between questions and tests"
        },
        {
            "name": "ConservativeDiagnosticAgent", 
            "description": "Asks many questions before ordering tests",
            "strategy": "Prefers detailed history taking over testing"
        },
        {
            "name": "AggressiveDiagnosticAgent",
            "description": "Orders many tests quickly",
            "strategy": "Rapidly orders comprehensive test panels"
        },
        {
            "name": "LLMDiagnosticAgent",
            "description": "Uses language models for intelligent decision-making",
            "strategy": "AI-powered diagnostic reasoning"
        }
    ]
    
    for agent in agents:
        print(f"• {agent['name']}")
        print(f"  Description: {agent['description']}")
        print(f"  Strategy: {agent['strategy']}")
        print()

def show_evaluation_metrics():
    """Display information about evaluation metrics."""
    print("Evaluation Metrics")
    print("=" * 18)
    print()
    
    print("Primary Metric: Diagnostic Accuracy")
    print("- Percentage of cases with judge score ≥ 4")
    print("- Formula: (Correct Cases) / (Total Cases)")
    print()
    
    print("Secondary Metric: Average Cumulative Cost")
    print("- Average total cost across all cases")
    print("- Includes physician visits ($300 each) and test costs")
    print()
    
    print("Judge Scoring Rubric (5-point Likert scale):")
    print("5 - Perfect/Clinically superior")
    print("4 - Mostly correct (minor incompleteness)")
    print("3 - Partially correct (major error)")
    print("2 - Largely incorrect")
    print("1 - Completely incorrect")
    print()

def show_usage_examples():
    """Display usage examples."""
    print("Usage Examples")
    print("=" * 15)
    print()
    
    print("1. Quick Test:")
    print("   python main.py quick")
    print()
    
    print("2. Single Case Demo:")
    print("   python main.py single")
    print()
    
    print("3. Full Benchmark:")
    print("   python main.py full")
    print()
    
    print("4. Interactive Demo:")
    print("   python main.py interactive")
    print()
    
    print("5. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    
    print("6. Set API Key:")
    print("   export OPENAI_API_KEY='your_api_key_here'")
    print()

def show_system_architecture():
    """Display system architecture information."""
    print("System Architecture")
    print("=" * 19)
    print()
    
    print("File Structure:")
    files = [
        "config.py - Configuration settings",
        "data_models.py - Pydantic data models", 
        "gatekeeper_agent.py - Gatekeeper agent implementation",
        "cost_estimator.py - Cost estimation module",
        "judge_agent.py - Judge agent implementation",
        "evaluation_protocol.py - Evaluation metrics and reporting",
        "sdbench.py - Main SDBench class",
        "synthetic_cases.py - Synthetic test cases",
        "example_agents.py - Example diagnostic agents",
        "main.py - Main script and demos",
        "test_system.py - System verification tests"
    ]
    
    for file in files:
        print(f"  • {file}")
    print()
    
    print("Key Features:")
    features = [
        "Sequential turn-based interactions",
        "Realistic clinical encounter simulation", 
        "Cost-aware evaluation with CPT code mapping",
        "Synthetic data generation for missing information",
        "Comprehensive 5-point diagnostic accuracy scoring",
        "Multiple agent types for comparison",
        "Interactive demo mode",
        "Performance visualization and reporting"
    ]
    
    for feature in features:
        print(f"  • {feature}")
    print()

def main():
    """Run the demo."""
    show_system_overview()
    show_synthetic_cases()
    show_agent_types()
    show_evaluation_metrics()
    show_usage_examples()
    show_system_architecture()
    
    print("SDBench Demo Complete!")
    print("=" * 22)
    print()
    print("To run the actual benchmark:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set API key: export OPENAI_API_KEY='your_key'")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main()
