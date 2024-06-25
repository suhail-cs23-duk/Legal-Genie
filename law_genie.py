
# pip install crewai

# pip install langchain_groq

# !pip install crewai crewai-tools

from langchain_groq import ChatGroq
from crewai import Agent , Task , Crew , Process
from crewai_tools import SerperDevTool , tool
from crewai_tools import ScrapeWebsiteTool
from langchain_community.tools.tavily_search import TavilySearchResults
import streamlit as st

llm = ChatGroq(model="mixtral-8x7b-32768",
           api_key="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")

# !pip install langchain-community

import os
os.environ["TAVILY_API_KEY"] = "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"
web_search_tool = TavilySearchResults()

web_search_tool = TavilySearchResults()

# web_search_tool.run("she murdered me , is it a crime")

@tool
def case_router_tool(question):
    """Router Function"""
    if 'criminal case' in question.lower():
        return 'criminal_case_search'
    elif 'civil case' in question.lower():
        return 'civil_case_search'
    elif 'corporate case' in question.lower():
        return 'corporate_case_search'
    elif 'international case' in question.lower():
        return 'international_case_search'
    else:
        return 'general_case_search'

# Example usage:
# case_router_tool("Find information on a recent criminal case.")
# Output: 'criminal_case_search'


Web_Lawyer = Agent(
  role='Web-Based Legal Researcher',
  goal='Collect information regarding legal cases in Indian context from the web.',
  backstory=(
    "You are a legal research expert specializing in sourcing detailed information about various legal cases in India context from the web."
    "Your primary focus is to gather comprehensive and accurate case information efficiently."
    "You excel in using web search tools to find relevant legal documents, case studies, and precedents."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[web_search_tool],
  llm=llm,
)


Retriever_Agent = Agent(
role="Retriever",
goal="Use the website links provided by Web_Lawyer and retrieve relevant information about the case from the web to answer the question",
backstory=(
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer."
),
verbose=True,
allow_delegation=False,
tools = [ScrapeWebsiteTool()],
llm=llm,
)

Grader_agent =  Agent(
  role='Answer Grader',
  goal='Filter out erroneous retrievals',
  backstory=(
    "You are a grader assessing relevance of a retrieved document to a user question."
    "If the document contains keywords related to the user question, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
  ),
  verbose=True,
  allow_delegation=False,
  llm=llm,
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked"
        "If the answer is relevant generate a clear and concise response."
        "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

case_router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords, decide whether it is related to a criminal case, civil case, international case, or corporate case."
    "Return a single word 'criminalcase' if it is related to a criminal case."
    "Return a single word 'civilcase' if it is related to a civil case."
    "Return a single word 'internationalcase' if it is related to an international case."
    "Return a single word 'corporatecase' if it is related to a corporate case."
    "If it does not match any of these categories, return 'websearch'."
    "Do not provide any other preamble or explanation."
    ),
    expected_output=("Give a single word response: 'criminalcase', 'civilcase', 'internationalcase', 'corporatecase', or 'websearch' based on the question."
    "Do not provide any other preamble or explanation."),
    agent=Web_Lawyer,
    tools=[case_router_tool]
)


retriever_task = Task(
    description=("Based on the response from the case_router_task extract information for the question {question} with the help of the respective tool."
    "Use the ScrapeWebsiteTool to scrape the web and use the web_search_tool to retrieve information from the website"
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "Return a clear and concise text as response."),
    agent=Retriever_Agent,
    context=[case_router_task],
   #tools=[retriever_tool],
)

grader_task = Task(
    description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=Grader_agent,
    context=[retriever_task],
)

hallucination_task = Task(
    description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
    "Respond 'yes' if the answer is in useful and contains fact about the question asked."
    "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=hallucination_grader,
    context=[grader_task],
)

answer_task = Task(
description=("Based on the response from the hallucination task for the question {question} evaluate whether the answer is useful to resolve the question."
                 " Perform a 'websearch' and return the considered response"),
    expected_output=("Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
                     "Perform a web search using 'web_search_tool' and return a clear and concise response"),
    context=[hallucination_task],
    agent=answer_grader,
    #tools=[answer_grader_tool],
)

rag_crew = Crew(
    agents=[Web_Lawyer, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
    tasks=[case_router_task,retriever_task, grader_task,hallucination_task,answer_task ],
    verbose=True,

)

def start(question):
    inputs = {"question": question} # This should be a dictionary with the question as a key
    result = rag_crew.kickoff(inputs=inputs)
    return result


st.set_page_config("LawGenie")
st.header("LawGenie")
st.info("Your Virtual Legal Expert")
user_question = st.text_area("Ask your questions")

if user_question:
    st.write(start(user_question))
