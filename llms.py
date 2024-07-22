from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

load_dotenv()
chat_llm = ChatOpenAI(model_name="qwen2-7b",
                      base_url="http://localhost:8000/v1",
                      temperature=0.1,
                      streaming=True)
