from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

# from langchain_tavily import TavilySearch
# from pydantic import BaseModel, Field
# from langchain.agents.structured_output import ToolStrategy
# from langchain.agents import create_agent

load_dotenv()

orders = [
    {
        "amount": 1299.99,
        "product_name": "PS5",
        "delivery_time": "50mins",
        "status": "on the way",
    },
    {
        "amount": 500,
        "product_name": "Laptop",
        "delivery_time": "1hr",
        "status": "deilvered",
    },
    {
        "amount": 200,
        "product_name": "iphone",
        "delivery_time": "1hr 30mins",
        "status": "deilvered",
    },
]


@tool
def get_order_details(product_name) -> dict:
    """
    Get compelete order based on product_name
    argument:
       product_name: "Used to get the product data from orders"
    returns:
       Get the product order and Returns a dict
    """
    for order in orders:
        if order["product_name"] == product_name:
            print(f"got order details {order['product_name']}")
            return order


@tool
def is_refund_eligible(order_data: dict) -> bool:
    """
     This tool check whether is user is eligible for refund based on the deilvery time and status
     argument:
         order_data: this contain the complete order data of a product with  deilvery_time , status and other info
    returns:
       This return a boolean user is eligible for refund or not
    """

    delivery_time = int(order_data["delivery_time"].replace("mins", ""))
    deilvery_status = order_data["status"]

    if delivery_time > 30 and deilvery_status == "on the way":
        print(
            f"eligible for refund since the estimate deilvery time is {order_data['delivery_time']} which is more than 30mins"
        )
        return True
    else:
        print(
            f"not eligible for refund since the estimate deilvery time is {order_data['delivery_time']} which is less than 30mins and status is {order_data['status']}"
        )
        return False


@tool
def get_refund(order_data: dict, is_refund_eligible: bool) -> str:
    """
     This tool provides the refund acmount for the eligible order
     argument:
         order_data: this contain the complete order data of a product with  deilvery_time , status and other info
         is_refund_eligible: this order is eligible for refund or not
    return :
        this returns refund amount
    """
    print(f"This order is eligible for {is_refund_eligible}")
    if is_refund_eligible:
        return f"The refund amount is {order_data['amount']}"
    else:
        return (
            f"The refund amount is {order_data['amount']}, since not eligible for refund"
        )


@traceable(name="Agent loopping")
def run_agent(query: str) -> str:

    MODEL = "qwen3:4b"
    tools = [get_order_details, is_refund_eligible, get_refund]
    chat = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm = chat.bind_tools(tools)

    tool_by_name = {tool.name: tool for tool in tools}

    messages = [
        SystemMessage(
            content="""
                 I am a shopping help assistance, based on my query get the order details and check whether i am eligible for refund or not if eligible refund the amount 
                 IMPORTANT STEPS DON'T IGNORE:
                 1. Get the order details based on product_name , call get_order_details tool first and get the complete order of that product 
                 2. Check if the product is eligible for refund or not , based on delivery_time and status use is_refund_eligible tool to get the result don't calculate manually call this tool only after get_order_details 
                 3. Get the refund amount , once u got the order details and is refund is eligible or not, use get_refund to get refund
           """
        ),
        HumanMessage(content=query),
    ]

    for i in range(1, 10, 1):
        ai_response = llm.invoke(messages)

        if not ai_response.tool_calls:
            print(ai_response)
            return ai_response.content

        messages.append(ai_response)
        execute_tool_call = ai_response.tool_calls[0]
        tool_call_id = execute_tool_call.get("id")
        tool_call_name = execute_tool_call.get("name")
        tool_call_agrument = execute_tool_call.get("args")

        selected_tool_call = tool_by_name.get(tool_call_name)
        execute_selected_tool_call = selected_tool_call.invoke(tool_call_agrument)

        messages.append(
            ToolMessage(
                content=str(execute_selected_tool_call), tool_call_id=tool_call_id
            )
        )
                   


def main() -> str:
    result = run_agent(
        "I have ordered PS5, can u check when i will get my order deilvered , i have waiting for more than 1 hour also check whether i am eligible for refund"
    )
    print(result)


if __name__ == "__main__":
    main()


# step 1 -> bind tools with llm
# step 2 -> invoke llm with the message -> ai message response with response_tool_call -> search
# step 3 -> append the ai result to the messages
# step 4 -> check if respone_tool_call is empty
# step 5 -> execute the tool call
# step 6 -> append the executed tool to messages
# step 7 -> pass all this to message list to llm -> repeats from step 2
