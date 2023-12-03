from langchain.chat_models import ChatOpenAI
from stuModel import StuModel
import sys
import time

sys.path.append("..")
from prompts.introPrompt import getIntroConversation
from prompts.recomdPrompt import getRecommendation, getCandid
from prompts.ragPrompt import getRAGQuery
from search.search_client import searchRAG
import json


def test_intro(student_info, llm, eval_llm):
    intro_scores = []
    for student in student_info:
        stu = StuModel(llm, eval_llm, student)
        profile = getIntroConversation(llm, stu)
        intro_score = stu.eval_profile(profile)
        print("Intro score: ", intro_score)
        intro_scores.append(intro_score)
        time.sleep(10)
    print("Total Intro score: ", sum(intro_scores) / len(intro_scores))


def test_recomd(student_info, llm, eval_llm):
    f = open("result_3.5.json", "w")
    f.write("[\n")
    hits = 0
    tots = 0
    for i, student in enumerate(student_info):
        stu = StuModel(llm, eval_llm, student)
        query = getRAGQuery(stu.getProfileWithoutCourse())
        lowercased_query = query.lower()
        is_graduate = "graduate" in lowercased_query or "grad" in lowercased_query or  "master" in lowercased_query
        level = "undergrad_collection"
        if (is_graduate):
            level = "grad_collection"
        course_context = searchRAG(query, level, 30)
        course_list = course_context.split("\n\n")
        course_names = [
            course.split(".)")[1].split(".")[0].strip()
            for course in course_list
            if ".)" in course
        ]
        recommed_names = getCandid(course_names, stu.getProfileWithoutCourse(), llm)

        candid_course_context = "\n\n".join(
            [
                course_list[i].split(".)")[1]
                for i in range(len(course_names))
                if course_names[i].split("-")[0] in recommed_names
            ]
        )
        recommed_list = getRecommendation(
            candid_course_context, stu.getProfileWithTakenCourse(), llm
        )
        hit, tot = stu.eval_recommd(recommed_list)
        hits += hit
        tots += tot

        profile = stu.getProfileDict()
        profile["recommendaion"] = recommed_list
        profile["score"] = hit / tot
        if i != 0:
            f.write(",\n")
        json.dump(profile, f, indent=2)

        # print(profile)

        time.sleep(10)
    print("Total Hit rate: ", hits / tots)

    f.write("\n]")
    f.close()


# intro
# gpt4-1106-preview 8.325
# gpt-3.5-turbo 6.937499999999999

# recomd
# 5 attempt in 3 actual courses
# gpt-4-1106-preview 1.13
# gpt-3.5-turbo 0.79
if __name__ == "__main__":
    student_info_path = "ba.json"
    with open(student_info_path, "r") as f:
        student_info = json.load(f)

    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    eval_llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    test_intro(student_info, llm, eval_llm)
    # test_recomd(student_info, llm, eval_llm)
