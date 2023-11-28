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
from langchain.evaluation import EvaluatorType, load_evaluator


def getIntroConversation(llm, student=None):
    student_schema = {
        "properties": {
            "interests": {"type": "string"},
            "goal": {"type": "string"},
            "experience": {"type": "string"},
            "course_taken": {"type": "string"},
            "extra_info": {"type": "string"},
        },
        "required": ["interests", "goal", "experience", "course_taken"],
    }

    stu_ext_chain = create_extraction_chain(schema=student_schema, llm=llm)

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """Act as an advisor at the University of Toronto. Engage in a short conversation to conduct an assessment of a student through questions to prepare for course recommendations. You need to explore the student's degree program, department, interests, academic goals, courses taken and research, volunteer, industry experience. Ask questions that encourage the student to share without feeling directly interrogated. DO NOT recommendate any courses. If the answer is ambiguous, you can also provide choices or suggestions to help the studnet answer in more detail.
If you gathered enough information for assessment, output in the following format:
Interest: {{student's interst}}
Academic Goal: {{student's academic goal}}
Experience: {{student's experience}}
Course Taken: {{courses the student took before}}
Extra Information: {{ extra information about the student}}
Otherwise, ask a new question to the student.
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
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)

    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    question = "Hi! Can you recommend a course for me?"

    context = ""
    while True:
        res = conversation({"question": question})

        criterion = {"question": "Does the output contain a question?"}
        evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria=criterion)
        eval_result = evaluator.evaluate_strings(prediction=res["text"], input=question)

        if eval_result["score"] == 0:
            peop = stu_ext_chain.run(res["text"])[0]
            context = peop
            context["summary"] = res["text"]
            print("Summary: ", context)
            break
        else:
            print("\nAdvisor: ", res["text"])
            if student is None:
                question = input("Input:\n")
            else:
                question = student.getResponse(res["text"])
    return context


if __name__ == "__main__":
    llm = ChatOpenAI(model_name="gpt-4-1106-preview")
    getIntroConversation(llm)
