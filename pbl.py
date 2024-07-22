import re
import json
import roles
import patient
import prompts
from llms import chat_llm
from pubmed import PubMedAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.openai import convert_message_to_dict


class Problem:
    def __init__(self, question):
        self._question = question
        self._answer = None
        self._discuss = dict()

    @property
    def question(self):
        return self._question

    @property
    def answer(self):
        return self._answer

    @answer.setter
    def answer(self, answer):
        self._answer = answer

    @property
    def discuss(self):
        return self._discuss

    @discuss.setter
    def discuss(self, name, opinion):
        self._discuss[name] = opinion


def convert_dialogue_to_str(dialogue):
    dialogue_str = ''
    for message in dialogue:
        message_dict = convert_message_to_dict(message)
        dialogue_str += f'{message_dict["role"]}: {message_dict["content"]}\n'
    return dialogue_str


def convert_discuss_to_str(problems):
    discuss_str = ''
    for problem in problems:
        discuss_str += f'问题：{problem.question}\n'
        for name, talk in problem.discuss.items():
            discuss_str += f'{name}: {talk}\n'
    return discuss_str


class PBL:
    def __init__(self, mock_patient=True):
        if mock_patient:
            self.patent_online = patient.MockPatient()
        self.nursing_intern = roles.NursingIntern()
        self.students = dict()
        for name, instruction in prompts.student_instructions.items():
            self.students[name] = roles.Student(instruction)
        self.inquiry_dialogue = list()
        self.medical_record = None
        self.problems = list()
        self.pubmed_api_wrapper = PubMedAPIWrapper()
        self.discuss_report = None
        self.nursing_plan = None

    def inquiry(self):
        intern_greet = self.nursing_intern.inquiry_greet()
        self.inquiry_dialogue.append(intern_greet)
        while not self.nursing_intern.inquiry_end:
            patient_response = self.patent_online.get_response(self.inquiry_dialogue[-1].content)
            self.inquiry_dialogue.append(HumanMessage(content=patient_response))
            intern_response = self.nursing_intern.inquiry_respond(patient_response)
            self.inquiry_dialogue.append(intern_response)

    def organize_records(self):
        system_message = SystemMessage(content=prompts.inquiry_organize_prompt)
        dialogue_str = convert_dialogue_to_str(self.inquiry_dialogue)
        user_message = HumanMessage(content=dialogue_str)
        self.medical_record = chat_llm([system_message, user_message]).content

    def raise_questions(self):
        prompt_template = PromptTemplate(input_variables=['medical_record'], template=prompts.raise_question_prompt)
        prompt = prompt_template.format(medical_record=self.medical_record)
        for name, student in self.students.items():
            result = student.get_response(prompt)
            question_match = re.search('\[.+?\]', result, re.DOTALL)
            if question_match is not None:
                questions = json.loads(question_match.group())
                self.problems.extend(Problem(q) for q in questions)

    def search_knowledge(self, question):
        search_prompt_template = PromptTemplate(input_variables=['question'], template=prompts.search_template)
        messages = [HumanMessage(content=search_prompt_template.format(question=question))]
        search_result = chat_llm(messages).content
        search_result_json = json.loads(re.search('\{.+\}', search_result, re.DOTALL).group())
        if isinstance(search_result_json['Pubmed'], str):
            query = search_result_json['Pubmed']
        else:
            query = search_result_json['Pubmed']['query']
        paper_results = self.pubmed_api_wrapper.run(query)
        return paper_results

    def panel_discuss(self):
        for problem in self.problems:
            discuss_prompt_template = PromptTemplate(input_variables=['medical_record', 'reference', 'question'], template=prompts.discuss_prompt)
            prompt = discuss_prompt_template.format(
                medical_record=self.medical_record, reference=problem.answer, question=problem.question)
            for name, student in self.students.items():
                result = student.get_response(prompt)
                problem.discuss[name] = result

    def summarize_discussion(self):
        system_message = SystemMessage(content=prompts.discuss_summarize_prompt)
        discuss_str = convert_discuss_to_str(self.problems)
        user_message = HumanMessage(content=discuss_str)
        self.discuss_report = chat_llm([system_message, user_message]).content

    def issue_plan(self):
        self.nursing_plan = self.nursing_intern.draft_plan()

    def start(self):
        self.inquiry()
        self.organize_records()
        self.raise_questions()
        for problem in self.problems:
            problem.answer = self.search_knowledge(problem.question)
        self.panel_discuss()
        self.summarize_discussion()
        self.issue_plan()


if __name__ == '__main__':
    pbl = PBL()
    pbl.start()
    print(pbl.nursing_plan)
