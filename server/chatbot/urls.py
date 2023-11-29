from django.urls import path
from .views import Chat, ChatReset, ChatUpload

app_name = "chatbot"

urlpatterns = [
    path("<str:user_input>", Chat.as_view(), name="main"),
    path("clear/", ChatReset.as_view(), name="clear"),
    path("upload/<path:file_path>", ChatUpload.as_view(), name="upload"),
]
