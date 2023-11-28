from dotenv import load_dotenv, find_dotenv
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import render
from django.http import JsonResponse
import os
from .gptView import Intro, RAGQuery, Recommend
from .search_client import searchRAG

from langchain.chat_models import ChatOpenAI

print(find_dotenv())
load_dotenv()

STATE = "INTRO"

MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')
OPENAI_KEY = os.getenv('OPENAI_KEY')

llm = ChatOpenAI(model_name=MODEL_NAME)

Course_ctx = ""
Student_ctx = ""

# Create your views here.
class Chat(APIView):
    throttle_scope = 'chatbot'
    # permission_classes = [IsAuthenticated,]
    
    def post(self, request, user_input):
        user = request.user
        print(user_input)
        # user_input = request.query_params['input']

        global STATE
        global Course_ctx
        global Student_ctx

        print(STATE)
        if STATE == "INTRO":
            res, end_flag = Intro(user_input, llm)
            if end_flag == True:
                print("End Intro!")
                print('Student information:', res)
                query = RAGQuery(res, llm)
                Course_ctx = searchRAG(query)
                print("searchRAG success!")
                Student_ctx = res
                user_input = "Hi! Can you recommend courses for me?"
                STATE = "RECOMMEND"
        
        if STATE == "RECOMMEND":
            res, recomend_flag = Recommend(user_input, Course_ctx, Student_ctx, llm)
            if recomend_flag == True:
                STATE = "END"

        print(res)
        return JsonResponse({"message": res})