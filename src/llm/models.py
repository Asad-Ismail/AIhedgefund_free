from __future__ import annotations
import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List
from pathlib import Path


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    GPT4FREE = "GPT4Free"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


# Load models from JSON file
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)
    
    models = []
    for model_data in models_data:
        # Convert string provider to ModelProvider enum
        provider_enum = ModelProvider(model_data["provider"])
        models.append(
            LLMModel(
                display_name=model_data["display_name"],
                model_name=model_data["model_name"],
                provider=provider_enum
            )
        )
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Load Ollama models from JSON
OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatOllama | ChatOpenAI | None:
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found.  Please make sure GROQ_API_KEY is set in your .env file.")
        return ChatGroq(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OPENAI:
        # Get and validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found.  Please make sure ANTHROPIC_API_KEY is set in your .env file.")
        return ChatAnthropic(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file.")
        return ChatDeepSeek(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found.  Please make sure GOOGLE_API_KEY is set in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OLLAMA:
        # For Ollama, we use a base URL instead of an API key
        # Check if OLLAMA_HOST is set (for Docker on macOS)
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
        )
    elif model_provider == ModelProvider.GPT4FREE:
        print(f"Using GPT4Free for LLM. {model_name}")
        from langchain_community.chat_models.openai import ChatOpenAI
        from pydantic import Field, BaseModel
        from g4f.client import AsyncClient, Client
        from g4f import Provider
        from g4f.Provider import RetryProvider,PollinationsAI
        from langchain_community.chat_models.openai import ChatOpenAI
        from langchain_core.runnables import Runnable
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
        from typing import Type, Any, Dict

        class ChatAI(ChatOpenAI):
            model_name: str = Field(default="gpt-4o", alias="model")
            
            @classmethod
            def validate_environment(cls, values: dict) -> dict:
                client_params = {
                    "api_key": values["api_key"] if "api_key" in values else None,
                    "provider": values["model_kwargs"]["provider"] if "provider" in values["model_kwargs"] else None,
                }
                values["client"] = Client(**client_params).chat.completions
                values["async_client"] = AsyncClient(**client_params).chat.completions
                return values
            
            def with_structured_output(
                self,
                schema: Type[BaseModel],
                method: str = "json_mode",
                **kwargs
            ) -> Runnable[Any, BaseModel]:
                """
                Custom structured output for GPT4Free that doesn't support native tool binding.
                Creates a chain that formats prompts and parses responses.
                """
                parser = PydanticOutputParser(pydantic_object=schema)
                
                def structured_chain(inputs):
                    # If inputs is a list of messages, convert to string
                    if isinstance(inputs, list):
                        # Extract the last human message content
                        prompt_content = ""
                        for msg in inputs:
                            if hasattr(msg, 'content'):
                                prompt_content += msg.content + "\n"
                    elif hasattr(inputs, 'content'):
                        prompt_content = inputs.content
                    else:
                        prompt_content = str(inputs)
                    
                    # Add JSON schema instructions to the prompt
                    enhanced_prompt = f"""{prompt_content}
                    Please respond with a valid JSON object that matches this schema:
                    {parser.get_format_instructions()}

                    Make sure your response is valid JSON wrapped in ```json``` tags."""
                    
                    # Call the original LLM
                    response = self.invoke(enhanced_prompt)
                    
                    # Parse the JSON response
                    try:
                        # Extract JSON from markdown if present
                        content = response.content
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            if json_end != -1:
                                json_content = content[json_start:json_end].strip()
                            else:
                                json_content = content[json_start:].strip()
                        else:
                            json_content = content
                        
                        # Parse using the Pydantic parser
                        return parser.parse(json_content)
                    except Exception as e:
                        print(f"Error parsing structured output: {e}")
                        # Return a default instance of the schema
                        return schema()
                
                # Create a runnable that applies the structured chain
                class StructuredRunnable(Runnable):
                    def invoke(self, input, config=None):
                        return structured_chain(input)
                    
                    def stream(self, input, config=None):
                        yield structured_chain(input)
                    
                    async def ainvoke(self, input, config=None):
                        return structured_chain(input)
                
                return StructuredRunnable()

        providers=RetryProvider([PollinationsAI], shuffle=False)
        return ChatAI(model=model_name,model_kwargs={"provider":providers})