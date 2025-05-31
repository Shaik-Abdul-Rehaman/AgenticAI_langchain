import streamlit as st
import pandas as pd
import re
from io import StringIO, BytesIO
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine, inspect
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

load_dotenv('.env')
st.set_page_config(page_title="Smart Data Assistant", page_icon="🧠")

# Layout
st.title("🔗 Smart Data Assistant with Dual Agents")
col1, col2 = st.columns(2)
api_key_query = col1.text_input("🔑 Groq API Key for Query Agent", type="password")
api_key_table = col2.text_input("🔑 Groq API Key for Table Agent", type="password")

# Guard clause
if not (api_key_query and api_key_table):
    st.info("Please enter both API keys to continue.")
    st.stop()

# LLMs
llm_query = ChatGroq(groq_api_key=api_key_query, model_name="llama-3.3-70b-versatile", temperature=0.1, streaming=True)
llm_table = ChatGroq(groq_api_key=api_key_table, model_name="llama-3.3-70b-versatile", temperature=0.1)

# Connect to DB
POSTGRES_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', 5432),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

try:
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
    )
    db = SQLDatabase(engine)
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# Functions
def get_schema():
    inspector = inspect(engine)
    schema_lines = []
    for table in inspector.get_table_names():
        cols = [f'"{c["name"]}" ({c["type"]})' for c in inspector.get_columns(table)]
        schema_lines.append(f"Table: {table}\nColumns: {', '.join(cols)}")
    return "\n\n".join(schema_lines)

def parse_text_to_table(text):
    try:
        if "|" in text:
            lines = [line.strip() for line in text.split("\n") if '|' in line and not line.strip().startswith('|-')]
            if lines:
                header, rows = lines[0], lines[1:]
                if rows:
                    cleaned = "\n".join([header] + rows)
                    df = pd.read_csv(StringIO(cleaned), sep="|", engine='python')
                    df.columns = [col.strip() for col in df.columns if col.strip()]
                    df.dropna(axis=1, how='all', inplace=True)
                    return df
        df = pd.read_csv(StringIO(text))
        return df if not df.empty else None
    except Exception as e:
        st.warning(f"⚠️ Parsing failed: {e}")
        return None

# Agent for DB queries
custom_prefix = f"""You are a SQL expert. Follow these rules strictly:
1. Only use tables: {', '.join(inspect(engine).get_table_names())}
2. Follow this schema:
{get_schema()}
3. Never make up table names or columns.
"""

toolkit = SQLDatabaseToolkit(db=db, llm=llm_query)

agent = create_sql_agent(
    llm=llm_query,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="generate",
    agent_kwargs={
        'prefix': custom_prefix,
        'max_execution_time': 15,
        'handle_parse_errors': "Check schema and try again"
    }
)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about the database!"}]

# Show schema info
with st.expander("📘 View Database Schema"):
    st.code(get_schema())

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Ask something about the database"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            raw_result = agent.run(prompt, callbacks=[st_callback])
            answer = raw_result.split("Final Answer:")[-1].strip() if "Final Answer:" in raw_result else raw_result
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write("🧠 **Query Result:**")
            st.write(answer)

            # Automatically send to 2nd agent (Table Converter)
            with st.spinner("🧮 Converting to table..."):
                table_prompt = f"Convert the following text into a markdown table format only:\n{answer}"
                output = llm_table.invoke(table_prompt)
                text_table = output.content if hasattr(output, 'content') else output
                df = parse_text_to_table(text_table)

                if df is not None and not df.empty:
                    st.success("📊 Table Generated from Answer")
                    st.dataframe(df)

                    # Downloads
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "converted_table.csv", "text/csv")
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    st.download_button("📥 Download Excel", excel_buffer.getvalue(), "converted_table.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("❌ Could not convert response to table format.")

        except Exception as e:
            err_msg = str(e)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            st.error(f"🔥 Error: {err_msg}")
