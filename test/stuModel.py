from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.evaluation import load_evaluator
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import json

W = {"interest": 0.3, "academic_goal": 0.3, "experience": 0.4}

class StuModel:
    def __init__(self, llm, profile):
        self.degree_program = profile["degree_program"]
        self.department = profile["department"]
        self.course_taken = "".join(profile["course_taken"].split(";")[:-2])
        self.course_to_take = profile["course_taken"].split(";")[-2:]
        self.interest = profile["interest"]
        self.academic_goal = profile["academic_goal"]
        self.experience = profile["experience"]
        self.extra_info = profile["extra_info"]

        print(self.degree_program)
        print(self.department)
        print(self.course_taken)
        print(self.course_to_take)
        print(self.interest)
        print(self.academic_goal)
        print(self.experience)
        print(self.extra_info)

        self.llm = llm
        self.memory = ConversationBufferMemory(return_messages=True)
        prompt = ChatPromptTemplate(
            input_variables=["input"],
            messages=[
                SystemMessagePromptTemplate.from_template(
                    f"You are a student in the University of Toronto. You're asked questions by advisors to help recommend course for you. You know nothing about the course provided this term or what course to take. Simply answer the question and do NOT provide other information.\nYour degree program: {self.degree_program}\n Your department: {self.department}\nYour interest: {self.interest}\nYour academic goal: {self.academic_goal}\nYour experience: {self.experience}\nYour took courses: {self.course_taken}\nYour extra information: {self.extra_info}"
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ],
        )

        self.chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter("history")
            )
            | prompt
            | self.llm
        )

    def getResponse(self, message):
        response = self.chain.invoke({"input": message})
        self.memory.save_context({"input": message}, {"output": response.content})
        print("\nStudent: ", response.content)
        return response.content

    def eval_profile(self, profile):
        print("profile", profile)
        accuracy_criteria = {
            "accuracy": """
Score 1: The answer is completely unrelated to the reference.
Score 3: The answer has minor relevance but does not align with the reference.
Score 5: The answer has moderate relevance but contains inaccuracies.
Score 7: The answer aligns with the reference but has minor errors or omissions.
Score 10: The answer is completely accurate and aligns perfectly with the reference."""
        }

        evaluator = load_evaluator(
            "labeled_score_string", criteria=accuracy_criteria, llm=self.llm
        )
        interest_result = evaluator.evaluate_strings(
            prediction=profile["interests"],
            reference=self.interest,
            input="What's the interest of the student?",
        )
        academic_goal_result = evaluator.evaluate_strings(
            prediction=profile["goal"],
            reference=self.academic_goal,
            input="what's the academic goal of the student?",
        )
        experience_result = evaluator.evaluate_strings(
            prediction=profile["experience"],
            reference=self.experience,
            input="what's the experience of the student?",
        )
        score = (
            interest_result["score"] * W["interest"]
            + academic_goal_result["score"] * W["academic_goal"]
            + experience_result["score"] * W["experience"]
        )
        return score

    def eval_course(self, course):
        pass
