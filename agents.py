import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tools import scrape_url, web_search

load_dotenv()

# Model setup with Groq API
# "llama-3.1-8b-instant" has a 30,000 TPM limit on Groq Free Tier (recommended to avoid 429 rate limits).
# "llama-3.3-70b-versatile" is also supported if you have sufficient quota.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.1,
    api_key=GROQ_API_KEY,
    max_retries=5,
    request_timeout=45,
)


# 1st agent: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are an AI search specialist. Your goal is to gather recent and reliable "
            "information about the user's research topic. Always call the web_search tool once "
            "with a focused query, then summarize key facts and include the URLs found in the results."
        ),
    )


# 2nd agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a web reader agent. You will receive search results that contain URLs. "
            "Pick ONE valid URL from the text and call the scrape_url tool with that exact URL. "
            "After receiving the scraped content, output a concise summary of the article."
        ),
    )


# 3rd chain: Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured, and insightful reports based on the gathered research.",
    ),
    (
        "human",
        """Write a comprehensive research report on the topic below using the gathered research.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (3 well-explained points with practical context)
- Deep Dive & Analysis
- Conclusion
- Sources (list all URLs found in the research)

Be factual, objective, and professional.""",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# 4th chain: Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest, objective, and specific in evaluating research reports.",
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
