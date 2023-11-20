from langchain.chat_models import ChatOpenAI
from stuModel import StuModel
import sys
sys.path.append("..")
from prompts.introPrompt import getIntroConversation
import json

student_info_path = 'student_info.json'
with open(student_info_path, 'r') as f:
    student_info = json.load(f)

scores = []
llm = ChatOpenAI(model_name="gpt-4-1106-preview")
for student in student_info:
    stu = StuModel(llm, student)
    profile = getIntroConversation(stu)
    score = stu.eval_profile(profile)
    print('Score: ', score)
    scores.append(score)

print('total score: ', sum(scores) / len(scores))