from prompts.introPrompt import getIntroConversation
from data_indexer.data_loader import load_database
from search.search_client import searchRAG;

def main():
    load_database()
    context = getIntroConversation()
    # Next call RAG search
    

if __name__ == "__main__":
    main()