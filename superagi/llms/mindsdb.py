from superagi.config.config import get_config
from superagi.lib.logger import logger
from superagi.llms.base_llm import BaseLlm
import mindsdb_sdk

class MindsDB(BaseLlm):
    def __init__(self, api_key, model=None, temperature=0.6, max_tokens=get_config("MAX_MODEL_TOKEN_LIMIT"), top_p=1,
                 frequency_penalty=0, presence_penalty=0, number_of_results=1, host="cloud.mindsdb.com"):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.number_of_results = number_of_results
        self.host = host
        self._server = None

    @property
    def server(self):
        if self._server is None:
            self._server = mindsdb_sdk.connect(login=self.api_key.split(':')[0],
                                             password=self.api_key.split(':')[1],
                                             host=self.host)
        return self._server

    def get_source(self):
        return "mindsdb"

    def get_api_key(self):
        return self.api_key

    def get_model(self):
        return self.model

    def get_models(self):
        try:
            models = self.server.list_models()
            return [model.name for model in models]
        except Exception as exception:
            logger.error(f"MindsDB Exception in get_models: {exception}")
            return []

    def verify_access_key(self):
        try:
            # Verify credentials by attempting to connect
            self.server
            return True
        except Exception as exception:
            logger.error(f"MindsDB Exception in verify_access_key: {exception}")
            return False

    def chat_completion(self, messages, max_tokens=None):
        try:
            if max_tokens is None:
                max_tokens = self.max_tokens

            # Format conversation history
            conversation = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                conversation += f"{role.capitalize()}: {content}\n"

            # Get model from MindsDB
            model = self.server.models.get(self.model)
            
            # Make prediction
            response = model.predict(conversation,
                                   options={
                                       'temperature': self.temperature,
                                       'max_tokens': max_tokens,
                                       'top_p': self.top_p,
                                   })

            # Format response to match expected structure
            return {
                "choices": [
                    {
                        "message": {
                            "content": response['response']
                        }
                    }
                ]
            }

        except Exception as exception:
            logger.error(f"MindsDB Exception in chat_completion: {exception}")
            return {
                "error": "ERROR",
                "message": str(exception)
            }