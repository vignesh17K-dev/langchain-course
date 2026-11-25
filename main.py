from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy


load_dotenv()


# # @tool
# def search(query: str) -> str:
#     """Tool that searches over internet
#     arguments:
#         query: The query to search for.
#     return:
#           The search results
#     """
#     print(f"Searching the internet for the information related to the query: {query}")
#     return client.search(query=query)


class AgentResponse(BaseModel):
    answer: str = Field(description="Detailed answer")
    summary: str = Field(description="Short summary")
    sources: list[str] = Field(description="Source URLs from search results")


llm = ChatOllama(model="qwen3:4b", temperature=0)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format = ToolStrategy(AgentResponse))


def main() -> str:
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for top 5 job postings for an full stack developer in  "
                )
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    main()
