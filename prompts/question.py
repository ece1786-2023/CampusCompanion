# ref: https://python.langchain.com/docs/use_cases/chatbots#conversation

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains import create_extraction_chain

llm = ChatOpenAI(model_name="gpt-4-1106-preview")

student_schema = {
    "properties": {
        "interests": {"type": "string"},
        "goal": {"type": "string"},
        "ability": {"type": "string"},
        "extra_info": {"type": "string"},
    },
    "required": ["interests", "goal", "ability"],
}

evaluate_schema = {
    "properties": {
        "summary": {"type": "string"},
        "talk": {"type": "string"},
        "finish_conversation": {"type": "boolean"},
    },
    "required": ["talk", "finish_conversation"],
}
stu_ext_chain = create_extraction_chain(schema=student_schema, llm=llm)
eval_ext_chain = create_extraction_chain(schema=evaluate_schema, llm=llm)


# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Act as an advisor of the Department of Electrical and Computer Engineering at the University of Toronto. You conduct an assessment of an ECE student through questions to prepare for course recommendations. Engage in a SHORT conversation that explores the student's interests, academic goals, ability, and courses taken. Ask questions that encourage the student to share without feeling directly interrogated. Do NOT make any recommendations. If the answer is ambiguous, you can also provide choices or suggestions to make it more specific.
If you gathered enough information for assessment, output in the following format:
Summary:
{{output the student's profile.}}
Talk:
End the conversation.
Otherwise, use the following format:
Summary:
{{summarize what new information you learned about the student from the answer to the system.}}
TALK:
{{keep chatting with the student.}}
"""
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
question = "Hi! Can you recommend a course for me?"
while True:
    res = conversation({"question": question})
    res = eval_ext_chain.run(res["text"])[0]

    if res["finish_conversation"] == True:
        peop = stu_ext_chain.run(res["summary"])[0]
        print(peop)
        break
    else:
        print("\nAdvisor: ", res["talk"])
        question = input("Input:\n")
