
from abc import ABC, abstractmethod
from google import genai
import requests


# Common interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Gemini cloud provider
class GeminiProvider(LLMProvider):

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text


# Ollama local provider
class OllamaProvider(LLMProvider):

    def __init__(self, model="qwen2.5-coder:3b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )

            response.raise_for_status()
            return response.json()["response"]

        except requests.exceptions.ConnectionError:
            return "Error: Ollama is not running. Start Ollama and try again."

        except requests.exceptions.HTTPError as e:
            return f"Ollama API Error: {e}"

        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out."


# Common client
class LLMClient:

    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)
