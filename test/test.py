from langchain.chat_models import ChatOpenAI
from stuModel import StuModel
import sys
import time

sys.path.append("..")
from prompts.introPrompt import getIntroConversation
from prompts.recomdPrompt import getRecommendation
from prompts.ragPrompt import getRAGQuery
import json

student_info_path = "student_info.json"
with open(student_info_path, "r") as f:
    student_info = json.load(f)

intro_scores = []
recom_scores = []
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
for student in student_info:
    stu = StuModel(llm, student)
    profile = getIntroConversation(llm, stu)
    intro_score = stu.eval_profile(profile)
    print("Intro score: ", intro_score)
    intro_scores.append(intro_score)

    course_context = getRAGQuery(str(student.course_taken))

    recommed_list = getRecommendation(course_context, str(student), llm)

    recom_score = stu.eval_recommd(recommed_list)
    print("Recommendation score: ", recom_score)
    recom_scores.append(recom_score)


print("Total Intro score: ", sum(intro_scores) / len(intro_scores))
print("Total Hit rate: ", sum(recom_scores) / len(recom_scores))
