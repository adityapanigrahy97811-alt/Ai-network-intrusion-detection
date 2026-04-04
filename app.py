import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from auth import login
from utils import save_log
from streamlit_autorefresh import st_autorefresh
import datetime

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Cyber Security Dashboard", layout="wide")

# ---------------------------
# LOAD CSS
# ---------------------------
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------
# MATRIX BACKGROUND (SUBTLE)
# ---------------------------
st.markdown("""
<style>
#matrix-bg {
    position: fixed;
    top: 0;
    left: 0;
    z-index: -1;
}
[data-testid="stAppViewContainer"] {
    background: transparent;
}
</style>

<canvas id="matrix-bg"></canvas>

<script>
const canvas = document.getElementById('matrix-bg');
const ctx = canvas.getContext('2d');

canvas.height = window.innerHeight;
canvas.width = window.innerWidth;

const letters = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const fontSize = 14;
const columns = canvas.width / fontSize;

const drops = [];
for (let x = 0; x < columns; x++) {
    drops[x] = 1;
}

function draw() {
    ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#38bdf8";
    ctx.font = fontSize + "px monospace";

    for (let i = 0; i < drops.length; i++) {
        const text = letters.charAt(Math.floor(Math.random() * letters.length));
        ctx.fillText(text, i * fontSize, drops[i] * fontSize);

        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
            drops[i] = 0;
        }

        drops[i]++;
    }
}

setInterval(draw, 33);
</script>
""", unsafe_allow_html=True)

# ---------------------------
# LOGIN SYSTEM
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("", ["Dashboard", "Logs", "Profile"])

# ---------------------------
# LOAD MODEL
# ---------------------------
try:
    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Actual Error: {e}")
    st.stop()


# ===================================================
# 📊 DASHBOARD
# ===================================================
if menu == "Dashboard":

    # AUTO REFRESH
    st_autorefresh(interval=5000, key="refresh")

    st.markdown("""
    <h1>🛡️ Network Intrusion Detection</h1>
    <p style='color:#94a3b8;'>Real-time AI cyber threat monitoring</p>
    """, unsafe_allow_html=True)

    st.markdown("🟢 **Live Monitoring Active**")
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV file",
        type=["csv"],
        key="upload1"
    )

    if uploaded_file is not None:
        try:
            # CHUNK PROCESSING (FAST)
            chunks = pd.read_csv(uploaded_file, chunksize=10000)

            results = []
            probabilities_all = []

            for chunk in chunks:
                chunk.columns = chunk.columns.str.strip()

                if "Label" in chunk.columns:
                    chunk = chunk.drop("Label", axis=1)

                chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
                chunk.fillna(0, inplace=True)
                chunk = chunk.select_dtypes(include=[np.number])

                chunk_scaled = scaler.transform(chunk)

                preds = model.predict(chunk_scaled)
                probs = model.predict_proba(chunk_scaled)[:, 1]

                results.extend(preds)
                probabilities_all.extend(probs)

            attack = int(np.sum(results))
            normal = len(results) - attack
            total = len(results)

            save_log(total, attack, normal)

            # METRICS
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Traffic", total)
            col2.metric("Normal Traffic", normal)
            col3.metric("Threats Detected", attack)

            st.markdown("---")
            st.markdown("### 📊 Traffic Analysis")

            col4, col5 = st.columns(2)

            with col4:
                fig1, ax1 = plt.subplots()
                ax1.pie([normal, attack], labels=["Normal", "Attack"], autopct="%1.1f%%")
                st.pyplot(fig1)

            with col5:
                fig2, ax2 = plt.subplots()
                ax2.bar(["Normal", "Attack"], [normal, attack])
                st.pyplot(fig2)

            # PROBABILITY GRAPH
            st.markdown("### 📉 Attack Probability Distribution")

            fig3, ax3 = plt.subplots()
            ax3.hist(probabilities_all, bins=30)
            st.pyplot(fig3)

            # ALERT
            if attack / total > 0.05:
                st.markdown(
                    "<h3 style='color:red;'>🚨 ALERT: HIGH THREAT DETECTED 🚨</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='color:lightgreen;'>✅ System Secure</h3>",
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {e}")

# ===================================================
# 🧾 LOGS PAGE
# ===================================================
elif menu == "Logs":

    import os

    st.title("📜 Attack Logs Dashboard")

    try:
        df = pd.read_csv("data/logs.csv")
        df.columns = ["Time", "Total Traffic", "Attacks", "Normal"]
        df["Time"] = pd.to_datetime(df["Time"])

        st.subheader("📊 Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Logs", len(df))
        col2.metric("Total Attacks", int(df["Attacks"].sum()))
        col3.metric("Traffic", int(df["Total Traffic"].sum()))

        st.markdown("---")

        st.line_chart(df.set_index("Time")[["Attacks", "Normal"]])
        st.bar_chart(df[["Attacks", "Normal"]])

        st.dataframe(df, use_container_width=True)

        # ---------------------------
        # DOWNLOAD BUTTON
        # ---------------------------
        st.download_button(
            "⬇ Download Logs",
            df.to_csv(index=False),
            file_name="logs.csv"
        )

        st.markdown("---")

        # ---------------------------
        # CLEAR LOGS 🔥
        # ---------------------------
        st.subheader("⚠️ Danger Zone")

        confirm = st.checkbox("I understand this will permanently delete all logs")

        if st.button("🗑 Clear Logs"):
            if confirm:
                if os.path.exists("data/logs.csv"):
                    os.remove("data/logs.csv")
                    st.success("✅ Logs cleared successfully!")
                    st.rerun()
                else:
                    st.warning("No log file found")
            else:
                st.error("Please confirm before deleting!")

    except:
        st.warning("No logs found yet")


# ===================================================
# 👤 PROFILE
# ===================================================
elif menu == "Profile":

    st.title("👤 Profile")

    st.write("Topic: Network Intrusion Detection")


    st.write("Name & Enrollment: Aditya , 160110523037")
    st.write("Name & Enrollment: Assis , 160110523034")
    st.write("Name & Enrollment: Tanish , 160110523095")
    st.write("Name & Enrollment: Sumit , 160110523056")
    st.write("Name & Enrollment: Krish , 160110523035")


    st.markdown("### 💻 language used")
    st.write("Python")
    st.write("Machine Learning")
    st.write("Data Analysis")           
