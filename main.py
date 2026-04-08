from dotenv import load_dotenv
from src.embeddings import get_embeddings
from src.vectorstore import load_vectorstore
from src.retriever import get_retriever
from src.prompt import get_prompt
from src.chain import get_chain
from src.guardrail import is_relevant_query, get_answer_with_sources
from config import CHROMA_DIR

load_dotenv()

def main():
    print("=== ML/DL Tutor Learning ===\n")

    #load all components
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings , CHROMA_DIR)
    retriever = get_retriever(vectorstore,embeddings)
    prompt = get_prompt()
    chain = get_chain(retriever , prompt)
    print("\n=== ML/DL Tutor Ready ===\n")
    print("Type 'exit' to quit.\n")

    #chat loop
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Guardrail check
            if not is_relevant_query(user_input):
                print("\nTutor: I am an ML/DL tutor. Please ask questions "
                      "related to Machine Learning or Deep Learning.\n")
                continue

            # Get answer
            answer = get_answer_with_sources(chain, user_input)
            print(f"\nTutor: {answer}\n")

    if __name__ == "__main__":
        main()
