import prompts
from llms import chat_llm
from langchain.schema import SystemMessage, HumanMessage


class NursingIntern:
    def __init__(self):
        self.dialogue_record = list()
        self.inquiry_end = False

    def inquiry_greet(self):
        system_message = SystemMessage(content=prompts.inquiry_system_prompt)
        self.dialogue_record.append(system_message)
        result = chat_llm(self.dialogue_record)
        self.dialogue_record.append(result)
        return result

    def inquiry_respond(self, patient_input):
        user_message = HumanMessage(content=patient_input)
        self.dialogue_record.append(user_message)
        result = chat_llm(self.dialogue_record)
        if '再见' in result.content:
            self.inquiry_end = True
        self.dialogue_record.append(result)
        return result

    def draft_plan(self):
        system_message = SystemMessage(content=prompts.draft_plan_prompt)
        result = chat_llm([system_message])
        return result.content


class Student:
    def __init__(self, instruction):
        self.system_message = SystemMessage(content=instruction)

    def get_response(self, prompt):
        user_message = HumanMessage(content=prompt)
        result = chat_llm([self.system_message, user_message])
        return result.content
