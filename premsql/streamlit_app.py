import os
from dotenv import load_dotenv
from premsql.playground import AgentServer
from premsql.agents import BaseLineAgent
from premsql.generators import Text2SQLGeneratorPremAI
from premsql.executors import ExecutorUsingLangChain
from premsql.agents.tools import SimpleMatplotlibTool

load_dotenv()

text2sql_model = Text2SQLGeneratorPremAI(
    model_name="gpt-4o", experiment_name="text2sql_model", type="test",
    premai_api_key=os.environ.get("PREMAI_API_KEY"),
    project_id=os.environ.get("PREMAI_PROJECT_ID")
)

analyser_plotter_model = Text2SQLGeneratorPremAI(
    model_name="gpt-4o", experiment_name="text2sql_model", type="test",
    premai_api_key=os.environ.get("PREMAI_API_KEY"),
    project_id=os.environ.get("PREMAI_PROJECT_ID")
)

# Enter your Database path here. Supported SQLite, Postgres, MySQL and an unique session name.
db_connection_uri = "sqlite:///chinook.db"  # Changed to a sample database
session_name = "my_session"

agent = BaseLineAgent(
    session_name=session_name,
    db_connection_uri=db_connection_uri,
    specialized_model1=text2sql_model,
    specialized_model2=analyser_plotter_model,
    executor=ExecutorUsingLangChain(),
    auto_filter_tables=False,
    plot_tool=SimpleMatplotlibTool()
)

# Create Streamlit app
def main():
    st.title("Text-to-SQL AI Assistant")
    query = st.text_input("Enter your natural language query:")
    
    if query:
        if query.startswith('/'):
            response = agent(query)
            st.write(response)
        else:
            st.error("Please start your query with /query, /analyse, or /plot")

if __name__ == "__main__":
    import streamlit as st
    main()
