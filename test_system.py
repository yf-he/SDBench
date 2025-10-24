"""Test script to verify SDBench system without external dependencies."""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data_models import CaseFile, AgentAction, ActionType
        print("✓ data_models imported successfully")
    except ImportError as e:
        print(f"✗ data_models import failed: {e}")
        return False
    
    try:
        from config import Config
        print("✓ config imported successfully")
    except ImportError as e:
        print(f"✗ config import failed: {e}")
        return False
    
    try:
        from synthetic_cases import get_all_synthetic_cases
        print("✓ synthetic_cases imported successfully")
    except ImportError as e:
        print(f"✗ synthetic_cases import failed: {e}")
        return False
    
    return True

def test_data_models():
    """Test data model creation and validation."""
    print("\nTesting data models...")
    
    try:
        from data_models import CaseFile, AgentAction, ActionType
        
        # Test CaseFile creation
        case = CaseFile(
            case_id="TEST_001",
            initial_abstract="A test case for validation.",
            full_case_text="This is a complete test case with detailed information.",
            ground_truth_diagnosis="Test diagnosis",
            publication_year=2024,
            is_test_case=True
        )
        print("✓ CaseFile created successfully")
        
        # Test AgentAction creation
        action = AgentAction(
            action_type=ActionType.ASK_QUESTIONS,
            content="What are the patient's symptoms?"
        )
        print("✓ AgentAction created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Data model test failed: {e}")
        return False

def test_synthetic_cases():
    """Test synthetic case generation."""
    print("\nTesting synthetic cases...")
    
    try:
        from synthetic_cases import get_all_synthetic_cases
        
        cases = get_all_synthetic_cases()
        print(f"✓ Generated {len(cases)} synthetic cases")
        
        for i, case in enumerate(cases):
            print(f"  Case {i+1}: {case.case_id} - {case.ground_truth_diagnosis}")
        
        return True
    except Exception as e:
        print(f"✗ Synthetic cases test failed: {e}")
        return False

def test_example_agents():
    """Test example agent creation."""
    print("\nTesting example agents...")
    
    try:
        from example_agents import RandomDiagnosticAgent, ConservativeDiagnosticAgent, AggressiveDiagnosticAgent
        
        # Test agent creation
        random_agent = RandomDiagnosticAgent("TestRandom")
        conservative_agent = ConservativeDiagnosticAgent("TestConservative")
        aggressive_agent = AggressiveDiagnosticAgent("TestAggressive")
        
        print("✓ Example agents created successfully")
        
        # Test agent reset
        random_agent.reset()
        conservative_agent.reset()
        aggressive_agent.reset()
        
        print("✓ Agent reset methods work")
        
        return True
    except Exception as e:
        print(f"✗ Example agents test failed: {e}")
        return False

def test_configuration():
    """Test configuration validation."""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        
        # Test config creation (without API key validation)
        config = Config()
        print("✓ Config created successfully")
        
        # Test config values
        assert config.PHYSICIAN_VISIT_COST == 300.0
        assert config.CORRECT_DIAGNOSIS_THRESHOLD == 4
        assert config.GATEKEEPER_MODEL == "gpt-4o-mini"
        assert config.JUDGE_MODEL == "gpt-4o"
        
        print("✓ Config values are correct")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_agent_actions():
    """Test agent action generation."""
    print("\nTesting agent actions...")
    
    try:
        from example_agents import RandomDiagnosticAgent
        from data_models import AgentAction, ActionType
        
        agent = RandomDiagnosticAgent("TestAgent")
        
        # Test action generation
        action = agent.get_next_action("Test case abstract", [])
        assert isinstance(action, AgentAction)
        assert action.action_type in [ActionType.ASK_QUESTIONS, ActionType.REQUEST_TESTS, ActionType.DIAGNOSE]
        assert len(action.content) > 0
        
        print("✓ Agent action generation works")
        
        return True
    except Exception as e:
        print(f"✗ Agent action test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("SDBench System Test")
    print("==================")
    
    tests = [
        test_imports,
        test_data_models,
        test_synthetic_cases,
        test_example_agents,
        test_configuration,
        test_agent_actions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! SDBench system is ready.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
