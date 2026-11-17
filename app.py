
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="IPL AI Analyst",
    page_icon="🏏",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #080d19;
    color: #ffffff;
}
section[data-testid="stSidebar"] {
    background: #111827;
}
h1, h2, h3 {
    color: #ffffff !important;
}
p, label {
    color: #d1d5db;
}
div.stButton > button {
    background: #f59e0b;
    color: #080d19;
    border-radius: 10px;
    font-weight: bold;
    border: none;
}
div.stButton > button:hover {
    background: #fbbf24;
    color: #080d19;
}
.metric-card {
    background: #151f32;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #26364f;
    margin-bottom: 10px;
}
.footer {
    text-align: center;
    color: #9ca3af;
    padding: 25px;
    margin-top: 40px;
    border-top: 1px solid #26364f;
}
.small {
    color: #9ca3af;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- IPL TEAMS ----------------

IPL_TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad"
]

TEAM_SHORT = {
    "Chennai Super Kings": "CSK",
    "Delhi Capitals": "DC",
    "Gujarat Titans": "GT",
    "Kolkata Knight Riders": "KKR",
    "Lucknow Super Giants": "LSG",
    "Mumbai Indians": "MI",
    "Punjab Kings": "PBKS",
    "Rajasthan Royals": "RR",
    "Royal Challengers Bengaluru": "RCB",
    "Sunrisers Hyderabad": "SRH"
}

# ---------------- HEADER ----------------

st.markdown("""
# 🏏 IPL AI ANALYST
### MACHINE LEARNING • CRICKET ANALYTICS • PREDICTIONS
""")

st.caption(
    "Explore IPL historical matches, team performance "
    "and machine learning predictions."
)

st.divider()

# ---------------- DATASET UPLOAD ----------------

st.sidebar.title("🏏 IPL Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload IPL Matches CSV",
    type=["csv"]
)

df = None
model = None
model_accuracy = None
feature_columns = []

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"CSV error: {e}")

# ---------------- DATA CLEANING ----------------

if df is not None:
    if "winner" in df.columns:
        df["winner"] = df["winner"].fillna("Unknown")

    if "team1" in df.columns:
        df["team1"] = df["team1"].fillna("Unknown")

    if "team2" in df.columns:
        df["team2"] = df["team2"].fillna("Unknown")

# ---------------- MODEL TRAINING ----------------

if df is not None:

    required = ["team1", "team2", "winner"]

    if all(col in df.columns for col in required):

        feature_columns = [
            c for c in [
                "team1",
                "team2",
                "venue",
                "toss_winner",
                "toss_decision"
            ]
            if c in df.columns
        ]

        training_data = df[
            feature_columns + ["winner"]
        ].copy()

        training_data = training_data.dropna()

        training_data = training_data[
            training_data["winner"].isin(
                pd.unique(training_data[["team1", "team2"]].values.ravel())
            )
        ]

        if (
            len(training_data) >= 30
            and training_data["winner"].nunique() >= 2
        ):

            X = training_data[feature_columns]
            y = training_data["winner"]

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )

                preprocessor = ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore"),
                            feature_columns
                        )
                    ]
                )

                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced"
                    ))
                ])

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                model_accuracy = accuracy_score(
                    y_test, predictions
                )

            except Exception:
                model = None

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Team Analytics",
        "🤖 Match Predictor",
        "💬 IPL Chatbot",
        "📁 Dataset Explorer"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "🏠 Dashboard":

    st.subheader("🏆 IPL Overview")

    if df is None:
        st.info(
            "Upload your IPL matches CSV from the sidebar "
            "to activate historical statistics."
        )

        st.markdown("""
        ### Welcome to IPL AI Analyst!

        Features available in this project:

        - Historical match analysis
        - Team performance statistics
        - Machine learning match prediction
        - Interactive cricket chatbot
        - Dataset exploration
        """)

    else:

        total_matches = len(df)

        total_teams = (
            pd.concat([
                df["team1"],
                df["team2"]
            ]).nunique()
            if "team1" in df.columns and "team2" in df.columns
            else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Matches", total_matches)
        col2.metric("Teams in Dataset", total_teams)

        if "season" in df.columns:
            col3.metric(
                "Seasons Covered",
                df["season"].nunique()
            )
        else:
            col3.metric("ML Model", "Ready" if model else "Pending")

        st.divider()

        if "winner" in df.columns:

            wins = (
                df[df["winner"] != "Unknown"]["winner"]
                .value_counts()
                .reset_index()
            )

            wins.columns = ["Team", "Wins"]

            st.subheader("🏆 Most Match Wins")

            fig = px.bar(
                wins.head(10),
                x="Team",
                y="Wins",
                color="Wins",
                template="plotly_dark",
                title="Team-wise Match Wins"
            )

            st.plotly_chart(fig, use_container_width=True)

        if model_accuracy is not None:
            st.metric(
                "Model Test Accuracy",
                f"{model_accuracy * 100:.2f}%"
            )

# ---------------- TEAM ANALYTICS ----------------

elif page == "📊 Team Analytics":

    st.subheader("📊 Team Performance Analytics")

    if df is None or "winner" not in df.columns:

        st.warning("Upload a valid IPL matches CSV first.")

    else:

        team_wins = (
            df[df["winner"] != "Unknown"]["winner"]
            .value_counts()
            .reset_index()
        )

        team_wins.columns = ["Team", "Wins"]

        st.dataframe(
            team_wins,
            use_container_width=True,
            hide_index=True
        )

        fig = px.pie(
            team_wins,
            names="Team",
            values="Wins",
            title="Share of Match Wins",
            template="plotly_dark",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

        if "season" in df.columns:

            st.subheader("Season-wise Matches")

            season_data = (
                df.groupby("season")
                .size()
                .reset_index(name="Matches")
            )

            fig2 = px.line(
                season_data,
                x="season",
                y="Matches",
                markers=True,
                template="plotly_dark"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

# ---------------- MATCH PREDICTOR ----------------

elif page == "🤖 Match Predictor":

    st.subheader("🤖 IPL Match Winner Prediction")

    if model is None:

        st.warning(
            "Prediction model is not ready. "
            "Upload a suitable matches CSV with at least "
            "30 valid matches and multiple winner classes."
        )

    else:

        available_teams = sorted(
            set(df["team1"].dropna().astype(str)) |
            set(df["team2"].dropna().astype(str))
        )

        if len(available_teams) < 2:

            st.warning("Not enough teams in the dataset.")

        else:

            col1, col2 = st.columns(2)

            with col1:
                team1 = st.selectbox(
                    "Select Team 1",
                    available_teams,
                    key="predict_team1"
                )

            with col2:
                team2 = st.selectbox(
                    "Select Team 2",
                    available_teams,
                    index=1,
                    key="predict_team2"
                )

            venue = None
            toss_winner = None
            toss_decision = None

            if "venue" in feature_columns:

                venues = sorted(
                    df["venue"].dropna().astype(str).unique()
                )

                venue = st.selectbox(
                    "Match Venue",
                    venues
                )

            if "toss_winner" in feature_columns:

                toss_winner = st.selectbox(
                    "Toss Winner",
                    [team1, team2]
                )

            if "toss_decision" in feature_columns:

                toss_decision = st.selectbox(
                    "Toss Decision",
                    ["bat", "field"]
                )

            if st.button("🔮 Predict Match Winner"):

                if team1 == team2:

                    st.error("Please select two different teams.")

                else:

                    input_data = {
                        "team1": team1,
                        "team2": team2
                    }

                    if "venue" in feature_columns:
                        input_data["venue"] = venue

                    if "toss_winner" in feature_columns:
                        input_data["toss_winner"] = toss_winner

                    if "toss_decision" in feature_columns:
                        input_data["toss_decision"] = toss_decision

                    input_df = pd.DataFrame(
                        [input_data],
                        columns=feature_columns
                    )

                    probabilities = model.predict_proba(input_df)[0]

                    classes = model.named_steps[
                        "classifier"
                    ].classes_

                    result_index = np.argmax(probabilities)

                    predicted_team = classes[result_index]

                    confidence = probabilities[result_index] * 100

                    st.success(
                        f"🏆 Predicted Winner: {predicted_team}"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "This is an experimental statistical "
                        "prediction, not a guaranteed result."
                    )

# ---------------- CHATBOT ----------------

elif page == "💬 IPL Chatbot":

    st.subheader("💬 Ask IPL AI Analyst")

    st.caption(
        "Ask questions about matches and team statistics "
        "in your uploaded dataset."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask something about IPL..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        answer = ""

        q = question.lower()

        if df is None:

            answer = (
                "Please upload an IPL matches CSV first. "
                "Then I can answer questions using its data."
            )

        elif "team1" in df.columns and "team2" in df.columns:

            if "total match" in q or "how many match" in q:

                answer = f"The dataset contains {len(df)} matches."

            elif "most win" in q or "top team" in q:

                if "winner" in df.columns:

                    top = (
                        df[df["winner"] != "Unknown"]
                        ["winner"]
                        .value_counts()
                    )

                    if not top.empty:
                        answer = (
                            f"The team with the most wins "
                            f"in this dataset is {top.index[0]} "
                            f"with {top.iloc[0]} wins."
                        )

            else:

                matched_team = next(
                    (
                        t for t in pd.unique(
                            pd.concat([
                                df["team1"],
                                df["team2"]
                            ]).dropna()
                        )
                        if str(t).lower() in q
                    ),
                    None
                )

                if matched_team is not None and "winner" in df.columns:

                    wins = (
                        df["winner"] == matched_team
                    ).sum()

                    played = (
                        (df["team1"] == matched_team) |
                        (df["team2"] == matched_team)
                    ).sum()

                    answer = (
                        f"📊 {matched_team}\n\n"
                        f"Matches played: {played}\n\n"
                        f"Matches won: {wins}"
                    )

                elif "season" in df.columns and "winner" in df.columns:

                    answer = (
                        "I can answer questions about total matches, "
                        "team wins and available seasons. "
                        "Try asking about a specific team."
                    )

                else:

                    answer = (
                        "Try asking about total matches, "
                        "top winning team or a team's match record."
                    )

        else:

            answer = (
                "Your CSV does not contain the required "
                "team1 and team2 columns."
            )

        if not answer:
            answer = "I could not find that information in the dataset."

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- DATASET EXPLORER ----------------

elif page == "📁 Dataset Explorer":

    st.subheader("📁 IPL Dataset Explorer")

    if df is None:

        st.info("Upload a CSV file to explore the dataset.")

    else:

        col1, col2 = st.columns(2)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        st.write("### Dataset Preview")

        st.dataframe(
            df.head(50),
            use_container_width=True
        )

        st.write("### Missing Values")

        missing = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing.columns = ["Column", "Missing Values"]

        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

        st.write("### Column Information")

        st.write(list(df.columns))

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <h3>🏏 IPL AI ANALYST</h3>
    <p>Machine Learning Based Cricket Analytics</p>
    <p><b>Created by Anant OP</b></p>
    <p class="small">
        Educational Machine Learning Project
    </p>
</div>
""", unsafe_allow_html=True)
