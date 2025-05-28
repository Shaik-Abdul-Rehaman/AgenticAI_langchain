import streamlit as st
import os
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, inspect
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

st.set_page_config(page_title="DB Query Agent", page_icon="🤖")
st.title("🤖 Natural Language Database Query Interface")

# Database credentials from .env (REQUIRED FIELDS)
POSTGRES_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', 5432),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# Validate all required credentials
missing = [k for k, v in POSTGRES_CONFIG.items() if not v]
if missing:
    st.error(f"🚨 Missing in .env file: {', '.join(missing)}")
    st.stop()

# Groq API setup (only user input)
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.info("🔑 Please enter your Groq API key to continue")
    st.stop()

# Database connection
def create_pg_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
        f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"
    )

try:
    engine = create_pg_engine()
    db = SQLDatabase(engine)
except Exception as e:
    st.error(f"🚨 Connection failed: {str(e)}")
    st.stop()

# Schema explorer function with proper sidebar integration
def schema_explorer():
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            st.sidebar.warning("No tables found in database")
            return
            
        selected_table = st.sidebar.selectbox(
            "📊 Database Tables", 
            tables,
            key="table_selector"
        )
        
        columns = inspector.get_columns(selected_table)
        column_names = [f'"{col["name"]}"' for col in columns]
        
        st.sidebar.markdown(f"**Columns in {selected_table}:")
        st.sidebar.code("\n".join(column_names))
        
    except Exception as e:
        st.sidebar.error(f"Error loading schema: {str(e)}")

# Modified get_schema() function
def get_schema():
    inspector = inspect(engine)
    table_list = inspector.get_table_names()
    schema = []
    for table in table_list:
        cols = [f'"{c["name"]}" ({c["type"]})' for c in inspector.get_columns(table)]
        fks = []
        for fk in inspector.get_foreign_keys(table):
            fks.append(f"→{fk['referred_table']}({','.join(fk['referred_columns'])})")
        
        schema_entry = [
            f"Table: {table}",
            f"Columns: {', '.join(cols)}",
            f"Relations: {', '.join(fks) if fks else 'None'}"
        ]
        schema.append("\n".join(schema_entry))
    
    return (
        "Available Tables:\n" + 
        "\n".join(table_list) + 
        "\n\nSchema Details:\n" + 
        "\n\n".join(schema)
    )




# LLM initialization
custom_prefix = f"""You are a SQL expert. Strict rules:
1. Use ONLY these tables: {', '.join(inspect(engine).get_table_names())}
2. Never reference non-existent tables
3. Follow schema:
{get_schema()}
4. For errors: Suggest checking table list"""


llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    streaming=True
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Updated agent configuration
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="generate",
    agent_kwargs={
        'prefix': custom_prefix,
        'max_execution_time': 15,
        'handle_parse_errors': "Check available tables and retry"
    }
)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything about the database!"}]

# Display schema explorer in sidebar before processing messages
schema_explorer()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Enter your query in natural language"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(prompt, callbacks=[st_callback])
            
            # Response processing
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
            else:
                final_answer = response

            if any(kw in final_answer.lower() for kw in ["[]", "no data", "empty"]):
                display_response = "No matching data found"
            else:
                display_response = final_answer

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.write("📊 *Result:*")
            st.write(final_answer)
            
        except Exception as e:
            error_msg = str(e)
            if any(err in error_msg for err in ["UndefinedColumn", "42703","relation does not exist"]):
                st.error("Schema mismatch! Available tables:")
                with st.expander("Database Schema"):
                    st.code(get_schema())
                response = "Requested table not found in database"
            else:
                response = f"Error: {error_msg.split(']')[0]}"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.error(response)