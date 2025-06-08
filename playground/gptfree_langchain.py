from langchain_community.chat_models.openai import ChatOpenAI
from pydantic import Field
from g4f.client import AsyncClient, Client
from g4f import Provider
from g4f.Provider import RetryProvider,PollinationsAI

## Check model and their providers here https://github.com/gpt4free/gpt4free.github.io/blob/main/docs/providers-and-models.md

# You can use any provider from the list above
providers=RetryProvider([PollinationsAI], shuffle=False)

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
    
if __name__ == "__main__":
    llm = ChatAI(model="deepseek-r1",model_kwargs={"provider":providers})
    print(llm.model_name)
    print(llm.invoke("What is the capital of France?"))