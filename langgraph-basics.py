from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph
from langchain.schema import HumanMessage
from pydantic import BaseModel
import json
# Load configurations from config.json
with open("config.json", 'r') as config_file:
    config = json.load(config_file)

AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = config["AZURE_OPENAI_ENDPOINT"]
DEPLOYMENT_NAME = config["DEPLOYMENT_NAME"]
# Azure OpenAI details

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY, 
    azure_endpoint=AZURE_OPENAI_ENDPOINT, 
    deployment_name=DEPLOYMENT_NAME, 
    model="gpt-4o",
    api_version="2024-02-01")

class ChatState(BaseModel):
    question: str
    answer: str | None = None

def generate_response(state: ChatState):
    response = llm.invoke([HumanMessage(content=state.question)])
    return ChatState(question=state.question, answer=response.content)

# Create a LangGraph workflow
workflow = StateGraph(ChatState)

# Define the starting node
workflow.add_node("ask_llm", generate_response)

# Define the edges (workflow path)
workflow.set_entry_point("ask_llm")

# Compile the graph
app = workflow.compile()

# Run the workflow with an input question
output = app.invoke(ChatState(question="What is LangGraph?"))
print(output)