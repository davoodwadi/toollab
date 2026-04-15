from openai import OpenAI, APIConnectionError
client = OpenAI(base_url='http://127.0.0.1:8081/v1', api_key='none')
try:
    client.models.list()
except APIConnectionError as e:
    print("Caught APIConnectionError!")
except Exception as e:
    print(f"Caught other exception: {type(e).__name__}")
