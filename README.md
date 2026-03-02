# 🛍️ Shopify AI Support Agent (LangGraph + Async Tool Calling)

An intelligent eCommerce customer support agent built using **LangGraph**, **LangChain**, and **OpenAI**, capable of:

- 🔄 Real-time Shopify order tracking
- 📚 FAQ & policy retrieval using FAISS vector search
- ⚡ Async tool execution with concurrent calls
- 🧠 Structured agent workflow using StateGraph
- 🎯 Intelligent routing between tools

---

## 🎥 Demo Video

▶️ Watch the full working demo here:  
**[Video Link]**

---

## 🚀 What This Project Demonstrates

### 1️⃣ Structured Agent Architecture
Built using **LangGraph StateGraph** instead of a basic agent loop.


---

### 2️⃣ Intelligent Tool Routing

The agent decides dynamically between:

- `order_status_tool` → For specific order tracking
- `faq_retriever_tool` → For general policy questions

It understands:
- Whether an order ID is provided
- Whether user is asking about *their* order
- When to ask for missing information

---

### 3️⃣ Real-Time Shopify Integration

Uses Shopify Admin API to:
- Fetch live order data
- Check payment status
- Check fulfillment status
- Return structured order details

Handles:
- Paid / Pending
- Fulfilled / Unfulfilled
- Missing orders
- Network failures

---

### 4️⃣ RAG (Retrieval-Augmented Generation)

- FAISS vector database
- OpenAI embeddings
- Retrieves top-K relevant FAQ documents
- Injects into tool response

Used for:
- Refund policy
- Delivery policy
- Process explanations
- Charges & fees

---

### 5️⃣ Fully Async Execution

- Async LLM calls
- Async HTTP requests (aiohttp)
- Concurrent tool execution with `asyncio.gather`
- Reduced response latency

---

## 🧠 System Prompt Logic

The assistant:

- Asks for order ID if missing
- Uses order tool only when ID is provided
- Uses retriever for general policy questions
- Responds in <100 words
- Maintains conversational tone

---

## 🏗️ Tech Stack

- **LangGraph**
- **LangChain**
- **OpenAI GPT-3.5**
- **FAISS**
- **aiohttp**
- **Python AsyncIO**
- **Shopify Admin API**

---
