from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import create_extraction_chain
from langchain.evaluation import EvaluatorType, load_evaluator
from langchain.chains import create_extraction_chain
from json import dumps

from .models import student_schema, course_schema
from .prompts import intro_prompt, rag_prompt, recommend_prompt


intro_memory = ConversationBufferMemory(
    memory_key="intro_history", return_messages=True, input_key="question"
)

recommend_memory = ConversationBufferMemory(
    memory_key="recommend_history", return_messages=True, input_key="question"
)


def Intro(question, llm, student=None):
    stu_ext_chain = create_extraction_chain(schema=student_schema, llm=llm)

    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(intro_memory.load_memory_variables)
            | itemgetter("intro_history")
        )
        | intro_prompt
        | llm
    )
    inputs = {"question": question}
    res = chain.invoke(inputs)

    intro_memory.save_context({"question": question}, {"output": res.content})

    criterion = {"question": "Does the output contain a summary of a student's interests, goals and experience?"}
    evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria=criterion)
    eval_result = evaluator.evaluate_strings(prediction=res.content, input=question)
    
    if eval_result["score"] == 1:
        # context = stu_ext_chain.run(res.content)[0]
        # context["summary"] = res.content
        # res = context
        res = res.content
        flag = True
    else:
        res = res.content
        flag = False

    return res, flag


def RAGQuery(content, llm):
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "user", "content": content},
    #         {
    #             "role": "assistant",
    #             "content": """Format a RAG search query to look for other similar courses based on the following summarization text, skills, interests, academic goals, and courses taken, exand to include other relevant terms. This will be a search query so only include relevant terms, it doesn't need to be a full sentence. The output MUST include department code
    #             Input: I am in History and I am interested in Roman history, and other ancient civilizations. I am also working part time as I am in a graduate program
    #             Output: Hist, Roman History, Ancient Civilizations, Ancient History, Ancient Greece, Roman Empire, Graduate
    #             Input:
    #             """,
    #         },
    #     ],
    #     temperature=0.9,
    #     max_tokens=256,
    #     top_p=0.75,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    # )
    chain = rag_prompt | llm
    input = {"content": content}
    res = chain.invoke(input)

    # print("Generated RAG search query:")
    # print(output)
    return res.content


def Recommend(question, course_context, student_context, llm):
    if type(course_context) is list:
        course_context = ",".join(course_context)
    elif type(course_context) is dict:
        course_context = dumps(course_context)
    if type(student_context) is dict:
        student_context = dumps(student_context)

    # using Stuff chain?
    # https://python.langchain.com/docs/modules/chains/document/stuff

    # qa_chain = load_qa_chain(OpenAI(model_name="gpt4-1106-preview"),  prompt=prompt, memory=memory)

    # qa_chain = LLMChain(llm=llm, prompt=prompt)
    course_ext_chain = create_extraction_chain(schema=course_schema, llm=llm)

    context = (
        "Course information:\n"
        + course_context
        + "\nStudent information:\n"
        + student_context
    )

    chain = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(recommend_memory.load_memory_variables)
            | itemgetter("recommend_history")
        )
        | recommend_prompt
        | llm
    )
    inputs = {"input_documents": context, "question": question}
    res = chain.invoke(inputs)
    recommend_memory.save_context({"question": question}, {"output": res.content})
    recommendation_list = []
    if "Success!" in res.content:
        recomd = res.content.split("Success!")[1]
        recommendation_list = course_ext_chain.run(recomd)
        res = dumps(recommendation_list)
        flag = True
    else:
        res = res.content.split("Fail!")[1]
        flag = False

    if recommendation_list is not []:
        print("Recommendation generated.")
        print("rank, code, name, score, reason")
        for i, course in enumerate(recommendation_list):
            print(
                f"{i+1}, {course['code']}, {course['name']}, {course['score']}, {course['reason']}"
            )

    return res, flag