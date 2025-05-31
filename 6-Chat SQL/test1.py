import streamlit as st
import pandas as pd
import re
from io import StringIO, BytesIO
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain.chat_models import AzureChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

# Load environment variables
load_dotenv()

# Set Streamlit config
st.set_page_config(page_title="Smart Data Assistant", page_icon="🧠")
st.title("🔗 Smart Data Assistant with Dual Agents")

# Azure OpenAI Config
AZURE_CONFIG = {
    'api_key': os.getenv("AZURE_OPENAI_API_KEY"),
    'api_version': os.getenv("AZURE_OPENAI_API_VERSION"),
    'azure_endpoint': os.getenv("AZURE_OPENAI_ENDPOINT"),
    'deployment_name': os.getenv("AZURE_OPENAI_DEPLOYMENT")
}

# LLM setup
llm = AzureChatOpenAI(
    openai_api_key=AZURE_CONFIG["api_key"],
    openai_api_version=AZURE_CONFIG["api_version"],
    azure_endpoint=AZURE_CONFIG["azure_endpoint"],
    deployment_name=AZURE_CONFIG["deployment_name"],
    temperature=0.1,
    streaming=True
)

# Database Config
POSTGRES_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', '5432'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Connect to database
try:
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
    )
    db = SQLDatabase(engine)
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")
    st.stop()

# Utility to display schema
@st.cache_data
def get_schema():
    inspector = inspect(engine)
    schema = []
    for table in inspector.get_table_names():
        columns = [f'"{col["name"]}" ({col["type"]})' for col in inspector.get_columns(table)]
        schema.append(f"Table: {table}\nColumns: {', '.join(columns)}")
    return "\n\n".join(schema)

# Parse markdown table or CSV text to DataFrame
def parse_text_to_table(text):
    try:
        if "|" in text:
            lines = [line.strip() for line in text.split("\n") if '|' in line and not line.strip().startswith('|-')]
            if lines:
                header, *rows = lines
                cleaned = "\n".join([header] + rows)
                df = pd.read_csv(StringIO(cleaned), sep="|", engine='python')
                df.columns = [col.strip() for col in df.columns if col.strip()]
                df.dropna(axis=1, how='all', inplace=True)
                return df
        df = pd.read_csv(StringIO(text))
        return df if not df.empty else None
    except Exception as e:
        st.warning(f"⚠️ Failed to parse table: {e}")
        return None

# Create the SQL agent
custom_prefix = f"""You are a SQL expert. Follow these rules strictly:
1. Use only these tables: {', '.join(inspect(engine).get_table_names())}
2. Refer to the following schema:
{get_schema()}
3. Do NOT make up table names or columns.
"""

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    early_stopping_method="generate",
    agent_kwargs={
        'prefix': custom_prefix,
        'max_execution_time': 15,
        'handle_parse_errors': "Check schema and try again."
    }
)

# Session state to track messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about the database!"}]

# Show schema
with st.expander("📘 View Database Schema"):
    st.code(get_schema())

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("table"):
            df = pd.DataFrame(msg["table"])
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "data.csv", "text/csv", key=f"csv_{hash(msg['content'])}")

            excel_buf = BytesIO()
            df.to_excel(excel_buf, index=False, engine='openpyxl')
            st.download_button("📥 Download Excel", excel_buf.getvalue(), "data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"excel_{hash(msg['content'])}")

# Input and response
if user_prompt := st.chat_input("Ask something about the database"):
    st.chat_message("user").write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        try:
            raw = agent.run(user_prompt, callbacks=[callback])
            answer = raw.split("Final Answer:")[-1].strip() if "Final Answer:" in raw else raw
            st.write("🧠 **Query Result:**")
            st.write(answer)

            with st.spinner("🔄 Converting result to table..."):
                table_prompt = f"Convert this into markdown table format:\n{answer}"
                output = llm.invoke(table_prompt)
                table_text = output.content if hasattr(output, 'content') else output
                df = parse_text_to_table(table_text)

                if df is not None and not df.empty:
                    df = df.iloc[1:]  # remove header repetition
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "result.csv", "text/csv")

                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    st.download_button("📥 Download Excel", excel_buffer.getvalue(), "result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    st.success("📊 Table generated successfully.")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "table": df.to_dict()})
                else:
                    st.write("⚠️ Could not parse table from response.")
                    st.session_state.messages.append({"role": "assistant", "content": answer + "\n\n⚠️ No table parsed."})

        except Exception as e:
            error_msg = f"🔥 Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
