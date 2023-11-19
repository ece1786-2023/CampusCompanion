from prompts.introPrompt import getIntroConversation
from prompts.recomdPrompt import getRecommendation
from data_indexer.data_loader import load_database
from search.search_client import searchRAG
import json

def main():
    load_database()
    student_context = getIntroConversation()
    # Next call RAG search
    course_context = searchRAG(student_context)

    # student_context and course_context are strings here.
    # TODO: replaced course context with a retrievaler
    recommendation_list = getRecommendation(course_context, student_context)
    json.dump(recommendation_list, open("result.json", "w"), indent=4)


if __name__ == "__main__":
    main()