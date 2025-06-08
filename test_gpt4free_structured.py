#!/usr/bin/env python3
from src.llm.models import get_model, ModelProvider
from pydantic import BaseModel

class TestOutput(BaseModel):
    message: str
    confidence: float

def test_gpt4free_structured():
    try:
        print("🧪 Testing GPT4Free with structured output...")
        
        # Test GPT4Free with structured output
        llm = get_model('gpt-4o-mini', ModelProvider.GPT4FREE)
        print("✅ GPT4Free model created successfully")
        
        structured_llm = llm.with_structured_output(TestOutput)
        print("✅ Structured output wrapper created successfully")
        
        result = structured_llm.invoke('Tell me about AI in a structured format with a confidence score')
        print(f"✅ GPT4Free structured output works!")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_gpt4free_structured() 