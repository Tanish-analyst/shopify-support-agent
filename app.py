from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Union
from langchain.tools import tool

import asyncio
import aiohttp
import os
import time

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


SHOPIFY_STORE = "tanishchatbot.myshopify.com"
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]


def load_retriever():
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        folder_path=".",
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 2})


retriever = load_retriever()


@tool
async def faq_retriever_tool(query: str) -> str:
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, retriever.invoke, query)

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


@tool
async def order_status_tool(order_id: str) -> str:
    order_id = order_id.strip()

    if not order_id.startswith("#"):
        order_id = f"#{order_id}"

    url = f"https://{SHOPIFY_STORE}/admin/api/2024-10/orders.json?status=any"

    headers = {
        "X-Shopify-Access-Token": ACCESS_TOKEN,
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return "Sorry, I couldn't connect to the order system right now."

            data = await response.json()

    for order in data.get("orders", []):
        if order.get("name") == order_id:
            customer = order.get("customer", {})
            items = order.get("line_items", [])

            financial_status = order.get("financial_status", "unknown")
            fulfillment_status = order.get("fulfillment_status") or "unfulfilled"

            if financial_status == "paid":
                payment_text = "✓ Payment completed"
            elif financial_status == "pending":
                payment_text = "⏳ Payment pending"
            else:
                payment_text = f"Payment: {financial_status}"

            if fulfillment_status == "fulfilled":
                shipping_text = "✓ Shipped/Delivered"
            elif fulfillment_status == "unfulfilled":
                shipping_text = "📦 Being prepared for shipment"
            else:
                shipping_text = f"Status: {fulfillment_status}"

            product_text = "\n".join(
                [f"  • {i['title']} (Qty: {i['quantity']})" for i in items]
            )

            return f"""Order {order.get('name')} Status:

{payment_text}
{shipping_text}

Products:
{product_text}

Total: ₹{order.get('total_price')}
Order Date: {order.get('created_at')[:10]}

Customer: {customer.get('first_name', '')} {customer.get('last_name', '')}
"""

    return f"I couldn't find order {order_id}. Please check your order number and try again."


tools = [faq_retriever_tool, order_status_tool]


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
).bind_tools(tools)


async def model_node(state: AgentState) -> AgentState:
    response = await llm.ainvoke(state["messages"])
    state["messages"].append(response)
    return state


def router(state: AgentState):
    last = state["messages"][-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tool"

    return END


async def tool_node_wrapper(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1]

    tasks = []

    for call in last_msg.tool_calls:
        tool_func = next(t for t in tools if t.name == call["name"])
        tasks.append(tool_func.ainvoke(call["args"]))

    results = await asyncio.gather(*tasks)

    for call, result in zip(last_msg.tool_calls, results):
        state["messages"].append(
            ToolMessage(
                content=str(result),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    return state


workflow = StateGraph(AgentState)

workflow.add_node("model", model_node)
workflow.add_node("call_tool", tool_node_wrapper)

workflow.set_entry_point("model")
workflow.add_conditional_edges("model", router)
workflow.add_edge("call_tool", "model")

app = workflow.compile()


SYSTEM_PROMPT = """
You are a Shopify App Store customer support assistant.

You help customers with:
1. Order Status Tracking (real-time)
2. FAQ & Policy Questions

RULES:

For Order-Specific Queries:
- If user mentions "my order", "my package" → They want THEIR specific order
- If they provide order ID (e.g., "#1005", "1005") → Use order_status_tool
- If NO order ID provided → Ask: "I'd be happy to check! Could you share your order ID? (e.g., #1001)"

After Order Status:
- PAID + UNFULFILLED → "Being prepared. Usually ships 1-2 days before delivery."
- FULFILLED → "Your order has been shipped/delivered!"
- PAYMENT PENDING → "Please complete payment to proceed."

For General Questions:
- Policies, processes, timelines → use faq_retriever_tool

Response Style:
- Concise and conversational
- Natural language
- Under 100 words unless detailed info needed

Tone: Friendly, empathetic ecommerce support.
"""


def interactive_conversation():
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Shopify Support Bot Ready...")
    print("=" * 50)

    while True:
        user = input("\nUser: ")

        if user.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye! 👋")
            break

        start_time = time.time()

        messages.append(HumanMessage(content=user))
        state = {"messages": messages}

        final = app.invoke(state)
        messages = final["messages"]

        ai_response = None

        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                ai_response = m.content
                break

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nAI: {ai_response}")
        print(f"\n⏱️  Response time: {elapsed_time:.2f} seconds")
        print("-" * 50)


if __name__ == "__main__":
    interactive_conversation()
