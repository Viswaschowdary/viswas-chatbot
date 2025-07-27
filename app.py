import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="💬 Falcon Chatbot", page_icon="🦅")

st.title("viswas Chatbot (RAG-ready)")
st.write("Chat with a powerful open LLM")

@st.cache_resource
def load_llm():
    model_id = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_llm()

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.chat_history.append({"user": user_input})
    response = generate_response(user_input)
    st.session_state.chat_history.append({"bot": response})

# Display history
for chat in st.session_state.chat_history:
    if "user" in chat:
        st.chat_message("user").write(chat["user"])
    else:
        st.chat_message("assistant").write(chat["bot"])
