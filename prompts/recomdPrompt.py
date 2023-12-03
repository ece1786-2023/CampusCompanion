from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains import create_extraction_chain
from json import dumps


llm = ChatOpenAI(model_name="gpt-4-1106-preview")

course_schema = {
    "properties": {
        "code": {"type": "string"},
        "name": {"type": "string"},
        "score": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["code", "name", "score", "reason"],
}

# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Act as an advisor at the University of Toronto. You select five courses from the provided course list as recommendations according to student's information. Do not choose a course taken before. Then, score them between 0 and 100 based on their alignment with the student's personal interests, relevance to their academic goals, and suitability to their experience. Finally, output the sorted list of courses and their corresponding scores. Do not ask questions or provide anything else.
At last, Output in the following format if you can make recommendations:
Success!
[1. {{course code #1}} {{course name #1}}; {{score #1}}; {{reason #1}}
2; {{course code #2}}; {{course name #2}}; {{score #2}}; {{reason #2}}
...
5; {{course code #5}}; {{course name #5}}; {{score #5}}; {{reason #5}}
]
Otherwise, use the following format:
Fail!
{{Other information you need to make recommendations.}}

{input_documents}

"""
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ],
    input_variables=["input_documents", "question"],
)


def getRecommendation(course_context, student_context, llm):
    if type(course_context) is list:
        course_context = ",".join(course_context)
    elif type(course_context) is dict:
        course_context = dumps(course_context)
    if type(student_context) is dict:
        student_context = dumps(student_context)

    # print(course_context)
    # print(student_context)
    # using Stuff chain?
    # https://python.langchain.com/docs/modules/chains/document/stuff
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key="question"
    )
    # qa_chain = load_qa_chain(OpenAI(model_name="gpt4-1106-preview"),  prompt=prompt, memory=memory)

    # qa_chain = LLMChain(llm=llm, prompt=prompt)
    course_ext_chain = create_extraction_chain(schema=course_schema, llm=llm)

    context = (
        "Course list:\n" + course_context + "\nStudent information:\n" + student_context
    )
    question = "Hi! Can you recommend courses for me?"
    recommendation_list = []
    while True:
        chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables)
                | itemgetter("chat_history")
            )
            | prompt
            | llm
            # | JsonOutputFunctionsParser()
        )
        inputs = {"input_documents": context, "question": question}
        res = chain.invoke(inputs)
        memory.save_context({"question": question}, {"output": res.content})
        if "Success!" in res.content:
            recomd = res.content.split("Success!")[1]
            recommendation_list = course_ext_chain.run(recomd)
            break
        else:
            talk = res.content.split("Fail!")[1]
            print("\nAdvisor: ", talk)
            question = input("Input:\n")

    print("Recommendation generated.")
    print("rank, code, name, score, reason")
    for i, course in enumerate(recommendation_list):
        print(
            f"{i+1}, {course['code']}, {course['name']}, {course['score']}, {course['reason']}"
        )

    return recommendation_list


candid_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Act as an advisor at the University of Toronto. You select {candid_size} courses from the provided course list as recommended course candidates based on their alignment with the student's personal interests, relevance to their goals, and suitability to their experience.
Output in the following format:
{{Course code #1 - Course name #1, Course code #2 - Course name #2, ..., Course code #{candid_size} - Course name #{candid_size}}}

{input_documents}

"""
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ],
    input_variables=["input_documents", "question", "candid_size"],
)


def getCandid(course_names, student_context, llm, candid_size=10):
    if type(course_names) is list:
        course_names = ", ".join(course_names)
    if type(student_context) is dict:
        student_context = dumps(student_context)

    context = (
        "Course lists:\n" + course_names + "\nStudent information:\n" + student_context
    )
    question = "Hi! Can you recommend courses for me?"
    chain = candid_prompt | llm
    inputs = {
        "input_documents": context,
        "question": question,
        "candid_size": candid_size,
    }
    res = chain.invoke(inputs)
    res = res.content
    return res


if __name__ == "__main__":
    course_context = """CSC301H1 - Introduction to Software Engineering
Hours: 24L/12T
An introduction to agile development methods appropriate for medium-sized teams and rapidly-moving projects. Basic software development infrastructure; requirements elicitation and tracking; estimation and prioritization; teamwork skills; basic modeling; design patterns and refactoring; discussion of ethical issues, and professional responsibility.
Prerequisite: CSC209H1, CSC263H1/ CSC265H1Exclusion: CSC301H5, CSCC01H3. NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC302H1 - Engineering Large Software Systems
Hours: 24L/12T
An introduction to the theory and practice of large-scale software system design, development, and deployment. Project management; advanced UML; reverse engineering; requirements inspection; verification and validation; software architecture; performance modelling and analysis.
Prerequisite: CSC301H1Exclusion: CSCD01H3. NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC303H1 - Social and Information Networks
Hours: 24L/12T
A course on how networks underlie the social, technological, and natural worlds, with an emphasis on developing intuitions for broadly applicable concepts in network analysis. Topics include: introductions to graph theory, network concepts, and game theory; social networks; information networks; the aggregate behaviour of markets and crowds; network dynamics; information diffusion; popular concepts such as "six degrees of separation," the "friendship paradox," and the "wisdom of crowds."
Prerequisite: CSC263H1/ CSC265H1/ CSC263H5/ CSCB63H3, STA247H1/ STA255H1/ STA257H1/ ECO227Y1/ STA237H1/ STAB52H3/ STAB57H3, MAT223H1/ MAT240H1Exclusion: CSCC46H3. NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC304H1 - Algorithmic Game Theory and Mechanism Design
Hours: 24L/12P
A mathematical and computational introduction to game theory and mechanism design. Analysis of equilibria in games and computation of price of anarchy. Design and analysis mechanisms with monetary transfers (such as auctions). Design and analysis of mechanisms without monetary transfers (such as voting and matching). This course is intended for economics, mathematics, and computer science students.
Prerequisite: STA247H1/ STA255H1/ STA257H1/ STA237H1/ PSY201H1/ ECO227Y1, ( MAT135H1, MAT136H1)/ MAT137Y1/ MAT157Y1Exclusion: NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Recommended Preparation: MAT223H1, CSC373H1Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC309H1 - Programming on the Web
Hours: 24L/12T
An introduction to software development on the web. Concepts underlying the development of programs that operate on the web; survey of technological alternatives; greater depth on some technologies. Operational concepts of the internet and the web, static client content, dynamic client content, dynamically served content, n-tiered architectures, web development processes, and security on the web. Assignments involve increasingly more complex web-based programs. Guest lecturers from leading e-commerce firms will describe the architecture and operation of their web sites.
Prerequisite: CSC209H1/ CSC209H5/ CSCB09H3/ ESC180H1/ ESC190H1/ CSC190H1/ (APS105H1, ECE244H1)Exclusion: CSC309H5, CSCC09H3. NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Recommended Preparation: CSC343H1Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC310H1 - Information Theory
Hours: 24L/12T
Measuring information. Entropy, mutual information and their meaning. Probabilistic source models and the source coding theorem. Data compression. Noisy channels and the channel coding theorem. Error correcting codes and their decoding. Applications to inference, learning, data structures and communication complexity.
Prerequisite: 60% or higher in CSC148H1, CSC263H1/ CSC265H1, MAT223H1/ MAT240H1Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)

CSC311H1 - Introduction to Machine Learning
Previous Course Number: CSC411H1
Hours: 24L/12T
An introduction to methods for automated learning of relationships on the basis of empirical data. Classification and regression using nearest neighbour methods, decision trees, linear models, and neural networks. Clustering algorithms. Problems of overfitting and of assessing accuracy. Basics of reinforcement learning.
Prerequisite: CSC207H1/ APS105H1/ APS106H1/ ESC180H1/ CSC180H1; MAT235Y1/​ MAT237Y1/​ MAT257Y1/​ (minimum of 77% in MAT135H1 and MAT136H1)/ (minimum of 73% in MAT137Y1)/ (minimum of 67% in MAT157Y1)/ MAT291H1/ MAT294H1/ (minimum of 77% in MAT186H1, MAT187H1)/ (minimum of 73% in MAT194H1, MAT195H1)/ (minimum of 73% in ESC194H1, ESC195H1); MAT223H1/ MAT240H1/ MAT185H1/ MAT188H1; STA237H1/ STA247H1/ STA255H1/ STA257H1/ STA286H1/ CHE223H1/ CME263H1/ MIE231H1/ MIE236H1/ MSE238H1/ ECE286H1Exclusion: CSC411H1, STA314H1, ECE421H1, CSC311H5, CSC411H5, CSCC11H3. NOTE: Students not enrolled in the Computer Science Major or Specialist program at A&S, UTM, or UTSC, or the Data Science Specialist at A&S, are limited to a maximum of 1.5 credits in 300-/400-level CSC/ECE courses.Recommended Preparation: MAT235Y1/ MAT237Y1/ MAT257Y1Distribution Requirements: Science
Breadth Requirements: The Physical and Mathematical Universes (5)
"""
    student_context = "A student in the Faculty of Arts & Science at the University of Toronto. Interested in computer science and mathematics. I am a hard-working student and I am willing to take challenging courses."
    res = getRecommendation(course_context, student_context)
    import json

    json.dump(res, open("recomd.json", "w"), indent=4)
