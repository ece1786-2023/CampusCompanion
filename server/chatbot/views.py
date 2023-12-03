from dotenv import load_dotenv, find_dotenv
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import render
from django.http import JsonResponse
import os
import json
import time
from .gptView import Intro, RAGQuery, Recommend, ExtractCourse
from .search_client import searchRAG

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader

print(find_dotenv())
load_dotenv(override=True)

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=OPENAI_API_KEY, verbose=True)


class SessionState:
    def __init__(self, session_id) -> None:
        self.session_id = session_id
        self.state = "INTRO"
        self.course_ctx = ""
        self.student_ctx = ""
        self.intro_memory = ConversationBufferMemory(
            memory_key="intro_history", return_messages=True, input_key="question"
        )
        self.recommend_memory = ConversationBufferMemory(
            memory_key="recommend_history", return_messages=True, input_key="question"
        )
        self.transcript = None


class SessionStatePool:
    def __init__(self) -> None:
        self.session_states = {}

    def get_session_state(self, session_id):
        if session_id not in self.session_states:
            print("new session state")
            self.session_states[session_id] = SessionState(session_id)
        return self.session_states.get(session_id, None)

    def set_session_state(self, session_id, state):
        self.session_states[session_id] = state

    def clear_session_state(self, session_id):
        if session_id in self.session_states:
            # self.session_states.pop(session_id)
            self.session_states[session_id] = SessionState(session_id)


session_state_pool = SessionStatePool()


# Create your views here.
class Chat(APIView):
    throttle_scope = "chatbot"
    # permission_classes = [IsAuthenticated,]

    def post(self, request, user_input):
        # bug not can run in local
        sess_id = request.session.session_key
        sess_state = session_state_pool.get_session_state(sess_id)

        state, course_ctx, student_ctx, intro_memory, recommend_memory, transcript = (
            sess_state.state,
            sess_state.course_ctx,
            sess_state.student_ctx,
            sess_state.intro_memory,
            sess_state.recommend_memory,
            sess_state.transcript,
        )

        print(user_input)
        print(state)
        print(intro_memory.load_memory_variables({}))

        if state == "INTRO":
            res, end_flag, intro_memory = Intro(user_input, llm, intro_memory)
            # End quiry
            if end_flag == True:
                query = RAGQuery(res, llm)
                lowercased_query = query.lower()
                # Check if "graduate" or "grad" is used
                is_graduate = "graduate" in lowercased_query or "grad" in lowercased_query or  "master" in lowercased_query
                level = "undergrad_collection"
                if (is_graduate):
                    level = "grad_collection"
                course_ctx = searchRAG(query, level)
                student_ctx = res
                state = "RECOMMEND"
                user_input = "Hi! Can you recommend courses for me?"
                # Add transcript into course_taken
                if transcript is not None:
                    stu = json.loads(student_ctx)
                    courses = ExtractCourse(transcript, query, llm)
                    stu["course_taken"] += courses
                    student_ctx = json.dumps(stu)

        if state == "RECOMMEND":
            res, recomend_flag, recommend_memory = Recommend(
                user_input, course_ctx, student_ctx, llm, recommend_memory
            )
            if recomend_flag == True:
                state = "END"

        sess_state.state = state
        sess_state.course_ctx = course_ctx
        sess_state.student_ctx = student_ctx
        sess_state.intro_memory = intro_memory
        sess_state.recommend_memory = recommend_memory
        session_state_pool.set_session_state(sess_id, sess_state)

        print(res)
        return JsonResponse({"message": res})


class ChatReset(APIView):
    def post(self, request):
        sess_id = request.session.session_key
        session_state_pool.clear_session_state(sess_id)
        # request.session.flush()
        return JsonResponse({"message": "Session cleared."})


class ChatUpload(APIView):
    def post(self, request, file_path):
        sess_id = request.session.session_key
        sess_state = session_state_pool.get_session_state(sess_id)

        print(file_path)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        docu = " ".join([page.page_content for page in pages])

        sess_state.transcript = docu
        session_state_pool.set_session_state(sess_id, sess_state)

        return JsonResponse({"message": "File uploaded."})
