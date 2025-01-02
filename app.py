import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

### Set up the Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver Using Google Gemma 2")

# Sidebar input for API key
groq_api_keys = st.sidebar.text_input("Groq API Key", type="password")

if not groq_api_keys:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# Initialize the LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_keys)

### Initializing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find information on the mentioned topics."
)

# Initialize the math tool
math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering math-related questions. Only input mathematical expressions."
)

# Define the reasoning tool prompt
prompt = """
You are a mathematical assistant. Solve the given problem step-by-step. Clearly explain each step and the reasoning behind it. Present the solution in the following structure:

1. Restate the problem.
2. Calculate the total runs scored in the first 10 overs.
3. Subtract the runs already scored from the target.
4. Calculate the required run rate for the remaining 40 overs.

Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine tools into a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions with step by step ."
)

### Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I am a Math Chatbot who can answer all your math questions!"}]

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

### Start the interaction
question = st.text_area("Enter your question:")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state["messages"].append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            try:
                response = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state["messages"].append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
