from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm
import requests

class TensorZero(BaseLlm):
    def __init__(self, api_key, model=None, temperature=0.6, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT"), top_p=1,
                 frequency_penalty=0, presence_penalty=0, number_of_results=1):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results

    def get_source(self):
        return "tensor_zero"

    def get_api_key(self):
        return self.api_key

    def get_model(self):
        return self.model

    def get_models(self):
        try:
            # Add TensorZero model listing API call here
            response = requests.get(
                "https://api.tensorzero.com/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code == 200:
                return response.json()["models"]
            return []
        except Exception as exception:
            logger.error(f"TensorZero Exception in get_models: {exception}")
            return []

    def verify_access_key(self):
        try:
            # Verify TensorZero API key is valid
            response = requests.get(
                "https://api.tensorzero.com/v1/verify",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.status_code == 200
        except Exception as exception:
            logger.error(f"TensorZero Exception in verify_access_key: {exception}")
            return False

    def chat_completion(self, messages, max_tokens=None):
        try:
            if max_tokens is None:
                max_tokens = self.max_tokens

            # Format messages for TensorZero API
            formatted_messages = []
            for message in messages:
                formatted_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })

            # Make TensorZero API call
            response = requests.post(
                "https://api.tensorzero.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": formatted_messages,
                    "temperature": self.temperature,
                    "max_tokens": max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "n": self.number_of_results
                }
            )

            if response.status_code == 200:
                completion = response.json()
                return {
                    "choices": [
                        {
                            "message": {
                                "content": completion["choices"][0]["message"]["content"]
                            }
                        }
                    ]
                }
            else:
                return {
                    "error": f"Error {response.status_code}",
                    "message": response.text
                }

        except Exception as exception:
            logger.error(f"TensorZero Exception in chat_completion: {exception}")
            return {
                "error": "ERROR",
                "message": str(exception)
            }