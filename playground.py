# playground.py


from phi.model.groq import Groq

from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.agent import Agent
import phi.api
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app


# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
phi.api=os.getenv("API_KEY")
# Define shared Groq model
groq_model = Groq(id="llama-3.2-1b-preview")  # ✅ Use the latest supported model

# Sub-agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

financial_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[financial_agent,web_search_agent]).get_app()



if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)
