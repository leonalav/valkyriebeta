"""
Minimal test to verify imports are working.
"""

try:
    import valkyrie_llm
    print("✅ Successfully imported valkyrie_llm")
except ImportError as e:
    print(f"❌ Failed to import valkyrie_llm: {e}")

try:
    from valkyrie_llm.model.reasoning import ChainOfThoughtReasoner
    reasoner = ChainOfThoughtReasoner()
    print("✅ Successfully imported and instantiated ChainOfThoughtReasoner")
except ImportError as e:
    print(f"❌ Failed to import ChainOfThoughtReasoner: {e}")
except Exception as e:
    print(f"❌ Error instantiating ChainOfThoughtReasoner: {e}")

print("Test completed.") 