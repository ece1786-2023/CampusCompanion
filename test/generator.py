from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
import json
import sys
import time
sys.path.append("..")
from prompts.ragPrompt import getRAGQuery
from search.search_client import searchRAG


class studentInfo(BaseModel):
    degree_program: str
    department: str
    interest: str
    academic_goal: str
    course_taken: str
    experience: str
    extra_info: str


examples = [
    {
        "example": """degree_program: Master of Engineering (MEng), department: Electrical and Computer Engineering, interest: Logic and Machine Learning, academic_goal: want to learn distributed machine learning system and want high grade, course_taken: Trustworthy Machine Learning and Introduction to Artificial Intelligence and Game Theory and Evolutionary Games, experience: Get bachelor's degree in computer engineering. Have a project to build a predictive models for public flow forecasts based on computer vision. Have an research experience in school's AI lab to build Event Tracking models on social media reviews using NLTK and Pytorch, extra_info: win the first prize in American Mathematical Competition."""
    },
    {
        "example": """degree_program: Master of Applied Science (MASc), department: Department of Computer Science, interest: Compilers and code generation, academic_goal: go deeper in computer system and architecture. Looking for a job in the field of computer system and architecture, course_taken: Compilers and Operating Systems and Topics in Database Management: Database System Technology, experience: BASc in Mechanical & Industrial Engineering and volunteered in student health education centre. Published papers in optimizing 3D printing, extra_info: won Dean's Honors List in 2018."""
    },
    {
        "example": """degree_program: Doctor of Philosophy (PhD), department: Department of Computer Science, interest: Human-AI interactions, academic_goal: graduate from Phd and learn more on art and design and usability with engineering. Also look for impactful opportunites to promote reaserch translation to the real-world, course_taken: Software engineering for machine learning and Numerical Methods for Optimization Problems and Neural Networks and Deep Learning, experience: BASc and MSc in Mechanical Engineering. Reaserch assistant in ML for Hearlthcare and another research for Human-Robot Interactions and and for Rehabilitation Engineering. Did internship at Amazon and Blackberry, extra_info: co-founded Marry Medical a startup that develop sensor-embedded IV catheter that won a National Award in Canada."""
    },
]

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")


def generate(num_runs=5, save_json=False, file_path="student_info.json"):
    llm = ChatOpenAI(model="gpt-4", temperature=1)

    prompt_template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject", "extra"],
        example_prompt=OPENAI_TEMPLATE,
    )

    synthetic_data_generator = create_openai_data_generator(
        output_schema=studentInfo,
        llm=llm,
        prompt=prompt_template,
    )

    synthetic_results = synthetic_data_generator.generate(
        subject="studentInfo",
        extra="Interest can include hobbies and subjects of interest and extracurricular activitiesand learning styles. Academic goals can include career goals and desired grades and learning objectives and future study plans. Courses_taken should include more than 4 courses taken in university. Experience can include work experience and research experience and volunteer experience.",
        runs=num_runs,
    )

    # print(synthetic_results)

    # transfer to real courses
    getCoursePrompt = PromptTemplate(
        input_variables=["course_list", "course_taken"],
        template="For each course in the target, find one course that is the same as or most similar to it in context. Do not output the original course in the target or anything else.\nOutput format: {{course_code #1}} {{course_name #1}}; {{course_code #2}} {{course_name #2}};...;{{course_code #n}} {{course_name #n}}\n\ntarget:\n{course_taken}\n\ncontext:\n{course_list}",
    )

    for stu in synthetic_results:
        query = getRAGQuery(stu.course_taken)
        course_list = searchRAG(query)
        chain = getCoursePrompt | llm
        formal_course_list = chain.invoke(
            {"course_taken": stu.course_taken, "course_list": course_list}
        )
        # print(formal_course_list)
        stu.course_taken = formal_course_list.content
        time.sleep(5)

    if save_json:
        serialized = [stu.dict() for stu in synthetic_results]
        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=2)

    return synthetic_results


if __name__ == "__main__":
    res = generate(save_json=True)
    print(res)
