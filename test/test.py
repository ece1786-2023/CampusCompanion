from langchain.chat_models import ChatOpenAI
from stuModel import StuModel
import sys
import time

sys.path.append("..")
from prompts.introPrompt import getIntroConversation
from prompts.recomdPrompt import getRecommendation
from prompts.ragPrompt import getRAGQuery
import json

student_info_path = "student_info_20.json"
with open(student_info_path, "r") as f:
    student_info = json.load(f)


def test_intro(llm, eval_llm):
    intro_scores = []
    for student in student_info:
        stu = StuModel(llm, eval_llm, student)
        profile = getIntroConversation(llm, stu)
        intro_score = stu.eval_profile(profile)
        print("Intro score: ", intro_score)
        intro_scores.append(intro_score)
        time.sleep(10)
    print("Total Intro score: ", sum(intro_scores) / len(intro_scores))


def test_recomd(llm, eval_llm):
    recom_scores = []
    for student in student_info:
        stu = StuModel(llm, student)
        course_context = getRAGQuery(str(student["course_taken"]))
        recommed_list = getRecommendation(course_context, str(student), llm)
        recom_score = stu.eval_recommd(recommed_list)
        print("Recommendation score: ", recom_score)
        recom_scores.append(recom_score)
        time.sleep(10)
    print("Total Hit rate: ", sum(recom_scores) / len(recom_scores))


# intro
# gpt4-1106-preview 7.674999999999999

if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    eval_llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    test_intro(llm, eval_llm)
    # test_recomd(llm, eval_llm)
