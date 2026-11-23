from dotenv import load_dotenv
from langchain_core.messages import content
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()


def main() -> str:
    print("hello world")

    information = """Grand Theft Auto VI is a 2026 action-adventure game developed and published by Rockstar Games. It is the eighth main entry in the Grand Theft Auto series, following 2013's Grand Theft Auto V, and the sixteenth entry overall. Set within the fictional US state of Leonida, based on Florida, and in the fictional city of Vice City, based on Miami, the story follows the criminal duo of Jason Duval and Lucia Caminos as they lead a life of crime to escape from corrupted police officers. Players are able to freely roam Leonida's open world.
                     Preliminary work began in 2014 following the release of Grand Theft Auto V, with principal production starting after the launch of Red Dead Redemption 2 in 2018. Rockstar's global development teams more than doubled since Red Dead Redemption 2. Miami and its surrounding areas were extensively researched for the game's world, which parodies Floridian and 2020s American culture. The protagonists' dynamic is inspired by Bonnie and Clyde, and the story was written to ensure players experience different areas of Vice City.
                     Following years of speculation and anticipation, Rockstar confirmed in February 2022 that the game was in development. That September, footage from unfinished versions was leaked. Grand Theft Auto VI was formally revealed in December 2023. It became the most preordered video game of all time, surpassing Cyberpunk 2077, with estimated sales exceeding 24 million digital copies, generating over US$2.3 billion in revenue prior to launch. Following two delays, further leaks, and an extensive marketing campaign, it was released on 19 November 2026 for the PlayStation 5 and Xbox Series X/S to universal critical acclaim. It tied with 1998's The Legend of Zelda: Ocarina of Time as the highest-rated video game on Metacritic, and it is the only game from the 21st century to achieve an overall score of 99."""

    template_summary = """refer the {information} for the queries i ask and answer my queries 
                       1. who is the two main characters name in the game 
                       2. when was the game release
                       """

    prompt = PromptTemplate(
        input_variables=["information"], template=template_summary
    )

    llm = ChatOllama( model="gemma3:latest", temperature=0)
    chain = prompt | llm
    result = chain.invoke({"information": information})
    print(result.content)


main()
