from boto3 import Session
from json import JSONDecodeError
from os import environ
import json

class Secrets:

    @staticmethod
    def apply_environment_secrets(
        secret_name: str | None = environ.get('APPLICATION'),
    ) -> None:
        if secret_name is None:
            raise ValueError('APPLICATION is not set in environment variables')

        aws = Session()
        secrets_manager = aws.client('secretsmanager')
        response: dict = secrets_manager.get_secret_value(SecretId=secret_name)

        try:
            secrets: dict | None = json.loads(response.get('SecretString'))
        
        except JSONDecodeError:
            raise ValueError('Secret value is not in JSON format')

        if secrets is None:
            raise ValueError('Secret value is empty')

        for secret, value in secrets.items():
            environ[secret] = value
