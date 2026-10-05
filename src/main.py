import sys
import logging

from .agent import agent

logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

def main():
    question = " ".join(sys.argv[1:]) or "Hello world!"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 50},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
