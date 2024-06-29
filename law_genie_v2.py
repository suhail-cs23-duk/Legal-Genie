# Live demo at https://legal-genie-v2.streamlit.app/

from crewai.tasks.task_output import TaskOutput
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from crewai import Agent , Task , Crew , Process
from crewai_tools import SerperDevTool , tool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

from langchain.chains import LLMChain
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

my_account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
my_api_token = st.secrets["CLOUDFLARE_API_KEY"]

llm = ChatGroq(model="mixtral-8x7b-32768", api_key=st.secrets["GROK_API_KEY"])

# llm = Ollama(model="llama3")
# llm = CloudflareWorkersAI(account_id=my_account_id, api_token=my_api_token, model="@cf/meta/llama-3-8b-instruct")

import os
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
search = TavilySearchAPIWrapper()
web_search_tool = TavilySearchResults(api_wrapper=search)

Retriever_Agent = Agent(
role="Retriever",
goal="Collect information regarding legal cases in Indian context from the web that is relevant to the question {question}.",
backstory=(
    "You are a legal research expert specializing in sourcing detailed information about various legal cases in India context from the web."
    "Your primary focus is to gather comprehensive and accurate case information efficiently."
    "You excel in using web search tools to find relevant legal documents, case studies, and precedents."
    "Use the keywords from the task to retrieve relevant information from the web. Use the 'web_search_tool' to achieve the task."
),
allow_delegation=False,
tools = [web_search_tool],
llm=llm,
)

Grader_agent =  Agent(
  role='Answer Grader',
  goal='Filter out erroneous retrievals',
  backstory=(
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
  ),
  allow_delegation=False,
  llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked"
        "If the answer is relevant generate a clear and detailed response."
        "If the answer generated is insufficient or irrelevant, exercise your knowledge of Indian legal frameworks to provide actionable advice. Avoid speculative content and prioritize realism and practicality."
    ),
    allow_delegation=False,
    llm=llm,
)

retriever_task2 = Task(
    description=("Use the 'web_search_tool' to scrape the web and return a list of websites and their content."
                 "Make sure to remove formatting from the content and only provide the text content of the website."
    ),
    expected_output=("List of websites and their content in the following format: "
                     "1. 1.1. URL of the website 1.2. Title of the website 1.3. Content of the website without the formatting\n2. 2.1. URL of the website 2.2. Title of the website 2.3. Content of the website without the formatting\n"),
    agent=Retriever_Agent,
)

grader_task = Task(
    description="Based on the response from the 'retriever_task2' for the question '{question}' evaluate whether the retrieved content is relevant to the question.",
    expected_output=("List of websites and their content in the following format: "
                     "1. 1.1. URL of the website 1.2. Title of the website 1.3. Content of the website without the formatting \n2. 2.1. URL of the website 2.2. Title of the website 2.3. Content of the website without the formatting\n"
                     "Provide a score for each website based on its relevance to the question asked."
                     "Sort the websites based on their relevance score."
                     "Remove irrelevant websites from the list."),
    agent=Grader_agent,
    context=[retriever_task2],
)

answer_task = Task(
    description=("Based on the content provided by the 'grader_task' for the question '{question}', generate a detailed and well put response."),
    expected_output=("A detailed and well put response."
                     "Ensure relevance and practicality even if the context provided is incomplete. "
                     "If context is insufficient or irrelevant, exercise your knowledge of Indian legal frameworks to provide actionable advice. Avoid speculative content and prioritize realism and practicality. "),
    context=[grader_task],
    agent=answer_grader,
    #tools=[answer_grader_tool],
)


def start(question):
    rag_crew = Crew(
        agents=[Retriever_Agent, Grader_agent, answer_grader],
        tasks=[retriever_task2, grader_task, answer_task],
        verbose=True,
    )

    inputs = {"question": question} # This should be a dictionary with the question as a key
    result = rag_crew.kickoff(inputs=inputs)
    return result

st.set_page_config("LawGenie")
st.header("LawGenie")
st.info("Your Virtual Legal Expert")
user_question = st.text_area("Ask your questions")

if user_question:
    st.write(start(user_question))
