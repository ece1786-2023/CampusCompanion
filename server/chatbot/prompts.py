from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

intro_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Act as an advisor at the University of Toronto. Engage in a short conversation to conduct an assessment of a student through questions to prepare for course recommendations. NEVER MAKE ANY RECOMMENDATION. You need to explore the student's degree program, department, interests, academic goals, courses taken and research, volunteer, industry experience. Ask questions that encourage the student to share without feeling directly interrogated. If the answer is ambiguous, you can also provide choices or suggestions to help the studnet answer in more detail. Do not ask for any further information.
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

rag_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Format a RAG search query to look for other similar courses based on the following summarization text, skills, interests, academic goals, and courses taken, exand to include other relevant terms. This will be a search query so only include relevant terms, it doesn't need to be a full sentence. The output MUST include department code
Input: I am in History and I am interested in Roman history, and other ancient civilizations. I am also working part time as I am in a graduate program
Output: Hist, Roman History, Ancient Civilizations, Ancient History, Ancient Greece, Roman Empire, Graduate
"""
        ),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

recommend_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            """Act as an advisor at the University of Toronto. You select five courses from the provided course list as recommended course candidates based on both student and course information. Next, you score all the course candidates between 0 and 100 based on their alignment with the student's personal interests, relevance to their academic goals, and suitability to their experience. Finally, output the sorted list of courses and their corresponding scores. Do not ask questions or provide anything else.
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