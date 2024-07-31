import os

from .evaluator import Evaluator
import re
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import requests
import time

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
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

        # api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        # if (not api_key):
        #     raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using openai evaluator.")

        # self.api_key = api_key
        # self.evaluator = ChatAnthropic(temperature=0, model="claude-3-5-sonnet-20240620")
        # # self.evaluator = ChatOpenAI(model=self.model_name,
        # #                             openai_api_key=self.api_key,
        # #                             **self.model_kwargs)

    def evaluate_response(self, response: str) -> int:
        # print('true answer: ', self.true_answer)
        # print('qustion: ', self.question_asked)
        print('evaluating')
        judge_template = [{ 'role': 'system', 'content': self.SYSTEM_MESSAGE},
                          {'role': 'user', 'content': "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. {criteria}\n[Ground truth]\n{reference}\nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: '[[rating]]', for example: 'Rating: [[5]]'.\n\n[Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n[The End of Assistant\'s Answer]".format(criteria=self.CRITERIA, reference=self.true_answer, input=self.question_asked, prediction=response)}]
        payload = {"temperature": 0, "messages": judge_template}
        openai_key = os.environ.get("OPENAI_KEY", None)
        openai_url = os.environ.get("OPENAI_URL", None)
        headers = {'api-key': openai_key, 'Content-Type': 'application/json'}
        url = openai_url
        i = 3
        while i > 0:
            try:
                verdict_response = requests.post(url=url, headers=headers, json=payload)
                verdict = verdict_response.json()['choices'][0]['message']['content']
                score = int(re.findall(r"\[\s*\+?(-?\d+)\s*\]", verdict)[0])
                i = 0
            except:
                i -= 1
                score = -1
                time.sleep(5)
                print(f'gpt call failed {i}')
        
            
        print('done evaluating')
        return score #int(eval_result['score'])

