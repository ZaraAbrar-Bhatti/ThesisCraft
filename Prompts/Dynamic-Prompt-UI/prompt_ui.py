from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st
from model import Google_Chat_Model
from descriptions import papers, msg, length, styles

with open(r"C:\Users\admin\Desktop\Gen_AI\Langchain\Prompts\Dynamic-Prompt-UI\styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_dotenv()
template = load_prompt("template.json")
LLM = Google_Chat_Model

st.title("📚 ThesisCraft")
st.markdown("Welcome to **ThesisCraft -a basic Research Tool**! Select any of the given below names to summarize the specific research paper.")

st.sidebar.title("About **ThesisCraft**")
st.sidebar.markdown(msg)

paper_input = st.selectbox( "Select Research Paper Name", papers )

style_input = st.selectbox( "Select Explanation Style", styles ) 

length_input = st.selectbox( "Select Explanation Length", length )

if st.button("Summarize"):
    chain = template | LLM
    result = chain.invoke({
    "paper_input": paper_input,
    "length_input": length_input,
    "style_input": style_input
})
    st.text(result.content)

st.markdown("---")
st.markdown("© 2025 Research Tool using Open Source Generative model. All rights reserved.")