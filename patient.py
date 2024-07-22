from llms import chat_llm
from langchain.schema import SystemMessage, HumanMessage


class MockPatient:
    def __init__(self):
        system_prompt = """
        你是一个模拟病人，你现在得了感冒，请根据相关医学知识回复合适的内容。
        """
        self.dialogue_record = list()
        self.dialogue_record.append(SystemMessage(content=system_prompt))

    def get_response(self, question):
        self.dialogue_record.append(HumanMessage(content=question))
        result = chat_llm(self.dialogue_record)
        self.dialogue_record.append(result)
        return result.content


class Patient:
    pass
