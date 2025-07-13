"""Helper functions for LLM"""

import json
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.graph.state import AgentState
from json_repair import repair_json
import json_repair
import time


def is_rate_limit_error(error: Exception) -> bool:
    """Check if the error is a rate limit error."""
    error_str = str(error).lower()
    return (
        "429" in error_str or 
        "rate limit" in error_str or 
        "quota" in error_str or
        "requests per minute" in error_str
    )

def calculate_backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 300.0) -> float:
    """
    Calculate exponential backoff delay with jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    
    Returns:
        Delay in seconds
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = base_delay * (2 ** attempt)
    
    # Cap at max_delay
    delay = min(delay, max_delay)
    
    # Add jitter (±25% randomness) to avoid thundering herd
    #jitter = delay * 0.25 * (2 * random.random() - 1)
    #delay += jitter
    
    # Ensure minimum delay
    return max(delay, base_delay)

def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    
    # Fallback to defaults if still not provided
    if not model_name:
        model_name = "gpt-4o"
    if not model_provider:
        model_provider = "OPENAI"

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)
    # For non-JSON support models or GPT4Free, we can use structured output
    # GPT4Free doesn't support native JSON mode but has custom structured output
    if not (model_info and not model_info.has_json_mode()) or model_provider == "GPT4FREE":
        print(f"Running stuctured wrapper!!!")
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    # We can provide list of monotically increasing delays for each retry the time increases, for now it is jsut 15s as 1 minute is normally enoguh for 
    # rate limit of gemini 
    delays = [15]

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if is_rate_limit_error(e):
                print(f"Detected Rate limit error backing off!")
                if attempt<len(delays):
                    base_delay=delays[attempt]
                else:
                    base_delay=delays[-1]

                # Calculate backoff delay
                delay = calculate_backoff_delay(attempt, base_delay, max_delay=300)
                
                if agent_name:
                    progress.update_status(
                        agent_name, 
                        None, 
                        f"Rate limit hit - waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                
                print(f"Rate limit detected. Backing off for {delay:.1f} seconds...")
                time.sleep(delay)

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        obj=json_repair.loads(content)
        return obj
    except Exception as e:
        print(f"Exception occured by content {content}")
        print(f"Error extracting JSON from response: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    """
    request = state.get("metadata", {}).get("request")

    if agent_name == 'portfolio_manager':
        # Get the model and provider from state metadata
        model_name = state.get("metadata", {}).get("model_name", "gpt-4o")
        model_provider = state.get("metadata", {}).get("model_provider", "OPENAI")
        return model_name, model_provider
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration
    model_name = state.get("metadata", {}).get("model_name", "gpt-4o")
    model_provider = state.get("metadata", {}).get("model_provider", "OPENAI")
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
