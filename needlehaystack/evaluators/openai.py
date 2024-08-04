import os

from .evaluator import Evaluator
import re
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
import anthropic
import requests
import time

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 2: The answer has minor relevance but does not align with the reference.
                Score 3: The answer has moderate relevance but contains inaccuracies.
                Score 4: The answer aligns with the reference but has minor omissions.
                Score 5: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}
    SYSTEM_MESSAGE = "You are a helpful assistant."
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked


    def evaluate_response(self, response: str) -> int:
        # print('true answer: ', self.true_answer)
        # print('qustion: ', self.question_asked)
        print('evaluating')
        print('true_answer: ', self.true_answer)
        prompt = {'role': 'user', 'content': "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}\n[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: '[[rating]]', for example: 'Rating: [[5]]'.\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]".format(criteria=self.CRITERIA, reference=self.true_answer, input=self.question_asked, prediction=response)}
        judge_template = [{ 'role': 'system', 'content': self.SYSTEM_MESSAGE}, prompt]
        # print('gpt4 template: ----------------------', judge_template)
        payload = {"temperature": 0, "messages": judge_template}
        openai_key = os.environ.get("OPENAI_KEY", None)
        openai_url = os.environ.get("OPENAI_URL", None)
        headers = {'api-key': openai_key, 'Content-Type': 'application/json'}
        url = openai_url
        try:
            verdict_response = requests.post(url=url, headers=headers, json=payload)
            verdict = verdict_response.json()['choices'][0]['message']['content']
            print(verdict)
            score = int(re.findall(r"\[\s*\+?(-?\d+)\s*\]", verdict)[0])
        except:
            print(verdict_response.text)
            print('gpt call failed, trying claude...')
            anthropic_key = os.environ.get("ANTHROPIC_KEY", None)
            client = anthropic.Anthropic(api_key=anthropic_key)
            try:
                verdict = client.messages.create(model="claude-3-5-sonnet-20240620", system=self.SYSTEM_MESSAGE, messages=[prompt], max_tokens=500).content[0].model_dump()['text']
            except:
                print('claude failed too')
                score = -1
        
            
        print('done evaluating')
        return score #int(eval_result['score'])

