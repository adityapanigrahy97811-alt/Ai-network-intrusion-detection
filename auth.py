import streamlit as st

# Dummy user (you can upgrade later)
USER = {
    "username": "admin",
    "password": "1234"
}

def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USER["username"] and password == USER["password"]:
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")