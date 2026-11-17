
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================================================
# IPL AI ANALYST
# Developed by Anant OP
# ==================================================

st.set_page_config(
    page_title="IPL AI Analyst",
    page_icon="🏏",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #080d19;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label {
    color: #cbd5e1;
}
.hero {
    background: linear-gradient(120deg,#152b48,#342052);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid #344968;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: 38px;
    margin: 0;
}
.hero p {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #131e30;
    border: 1px solid #293b55;
    padding: 15px;
    border-radius: 12px;
}
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    margin-top: 40px;
    border-top: 1px solid #293b55;
}
.note {
    color: #94a3b8;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TEAM INFORMATION ----------------

TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#D71920",
    "Royal Challengers Bangalore": "#D71920",
    "Kolkata Knight Riders": "#6736A5",
    "Sunrisers Hyderabad": "#F26522",
    "Rajasthan Royals": "#EA1A85",
    "Delhi Capitals": "#17479E",
    "Punjab Kings": "#ED1B24",
    "Kings XI Punjab": "#ED1B24",
    "Gujarat Titans": "#1B2133",
    "Lucknow Super Giants": "#00AEEF"
}

# ---------------- DATA CLEANING ----------------

def clean_data(data):
    data = data.copy()

    data.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in data.columns
    ]

    aliases = {
        "team_1": "team1",
        "team_2": "team2",
        "team_1_name": "team1",
        "team_2_name": "team2",
        "match_winner": "winner",
        "winning_team": "winner",
        "toss_winning_team": "toss_winner",
        "toss_decision": "toss_decision",
        "match_date": "date",
        "match_venue": "venue",
        "player_of_the_match": "player_of_match"
    }

    data = data.rename(columns=aliases)

    return data


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return clean_data(pd.read_csv(uploaded_file))

    try:
        return clean_data(pd.read_csv("ipl_matches_data.csv"))
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as error:
        st.error(f"Dataset loading error: {error}")
        return pd.DataFrame()


# ---------------- SIDEBAR ----------------

st.sidebar.title("🏏 IPL AI ANALYST")
st.sidebar.caption("Cricket Intelligence Dashboard")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload IPL Dataset (CSV)",
    type=["csv"]
)

df = load_dataset(uploaded_file)

st.sidebar.divider()
st.sidebar.markdown("### PROJECT DETAILS")
st.sidebar.write("**Developer:** Anant OP")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**Framework:** Streamlit")
st.sidebar.write("**ML Algorithm:** Random Forest")
st.sidebar.write("**Project:** IPL Match Analytics")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🏏 IPL AI ANALYST</h1>
    <p>Machine Learning Based Cricket Analytics & Match Prediction</p>
    <p class="note">Historical Data • Team Intelligence • AI Chatbot</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DATA VALIDATION ----------------

required = ["team1", "team2", "winner"]

valid_dataset = (
    not df.empty
    and all(col in df.columns for col in required)
)

if valid_dataset:

    for col in required:
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=required)

    df = df[
        (df["winner"] == df["team1"]) |
        (df["winner"] == df["team2"])
    ].copy()

    teams = sorted(
        set(df["team1"].dropna().astype(str)) |
        set(df["team2"].dropna().astype(str))
    )

else:
    teams = []

# ---------------- MODEL TRAINING ----------------

model = None
model_accuracy = None
model_error = None

feature_columns = []

if valid_dataset and len(df) >= 30:

    possible_features = [
        "team1",
        "team2",
        "venue",
        "toss_winner",
        "toss_decision"
    ]

    feature_columns = [
        col for col in possible_features
        if col in df.columns
    ]

    # Team and toss information must be available
    # before prediction. Venue is optional.

    training_data = df[
        feature_columns + ["winner"]
    ].copy()

    training_data = training_data.dropna(
        subset=["team1", "team2", "winner"]
    )

    # Keep only rows with a valid winner
    training_data = training_data[
        training_data["winner"].isin(
            pd.unique(
                training_data[["team1", "team2"]].values.ravel()
            )
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

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ])

            preprocessor = ColumnTransformer([
                ("categorical", categorical_pipeline, feature_columns)
            ])

            model = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced"
                ))
            ])

            model.fit(X_train, y_train)

            test_predictions = model.predict(X_test)

            model_accuracy = accuracy_score(
                y_test,
                test_predictions
            )

        except Exception as error:
            model_error = str(error)

# ---------------- DASHBOARD NAVIGATION ----------------

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📊 Team Analytics",
        "🤖 Match Predictor",
        "💬 IPL Chatbot",
        "📁 Dataset Explorer"
    ]
)

# ==================================================
# 1. DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.subheader("📊 IPL Overview")

    if not valid_dataset or df.empty:

        st.warning(
            "Dataset not found or required columns are missing. "
            "Upload a valid IPL CSV using the sidebar."
        )

        st.markdown("""
        ### Project Features

        1. Historical IPL Match Analytics
        2. Team Performance Comparison
        3. Machine Learning Match Prediction
        4. Interactive IPL Chatbot
        5. Dataset Exploration
        """)

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Matches", f"{len(df):,}")
        c2.metric("Teams", len(teams))

        if "season" in df.columns:
            c3.metric(
                "Seasons",
                df["season"].nunique()
            )
        else:
            c3.metric("Data Columns", len(df.columns))

        if model_accuracy is not None:
            c4.metric(
                "Model Test Accuracy",
                f"{model_accuracy * 100:.1f}%"
            )
        else:
            c4.metric("ML Model", "Not Ready")

        st.divider()

        st.subheader("🏆 Top Winning Teams")

        wins = (
            df["winner"]
            .value_counts()
            .reset_index()
        )

        wins.columns = ["Team", "Wins"]

        fig = px.bar(
            wins.head(10),
            x="Team",
            y="Wins",
            color="Team",
            color_discrete_map=TEAM_COLORS,
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor="#080d19",
            plot_bgcolor="#080d19",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        if model is not None:
            st.success("Machine Learning model trained successfully.")
        else:
            st.info("Machine Learning model is not ready yet.")

# ==================================================
# 2. TEAM ANALYTICS
# ==================================================

elif page == "📊 Team Analytics":

    st.subheader("📊 Team Performance Analytics")

    if not valid_dataset or df.empty:

        st.warning("Upload a valid dataset first.")

    else:

        wins = (
            df["winner"]
            .value_counts()
            .reset_index()
        )

        wins.columns = ["Team", "Wins"]

        st.write("### Team-wise Wins")

        st.dataframe(
            wins,
            use_container_width=True,
            hide_index=True
        )

        fig = px.pie(
            wins,
            names="Team",
            values="Wins",
            title="Share of Match Wins",
            hole=0.45,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        if "season" in df.columns:

            st.write("### Season-wise Matches")

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

        if "toss_winner" in df.columns:

            st.write("### Toss Statistics")

            toss_data = df.dropna(
                subset=["toss_winner"]
            )

            toss_wins = (
                toss_data["toss_winner"] ==
                toss_data["winner"]
            ).sum()

            if len(toss_data) > 0:
                st.metric(
                    "Toss Winner Also Won Match",
                    f"{toss_wins / len(toss_data) * 100:.1f}%"
                )

# ==================================================
# 3. MATCH PREDICTOR
# ==================================================

elif page == "🤖 Match Predictor":

    st.subheader("🤖 Machine Learning Match Predictor")

    if model is None:

        st.warning(
            "The prediction model could not be trained. "
            "Upload a dataset containing at least 30 valid "
            "matches and multiple winning teams."
        )

        if model_error:
            st.caption(f"Model information: {model_error}")

    else:

        col1, col2 = st.columns(2)

        with col1:
            team1 = st.selectbox(
                "Select Team 1",
                teams
            )

        with col2:
            team2 = st.selectbox(
                "Select Team 2",
                teams,
                index=1 if len(teams) > 1 else 0
            )

        input_values = {
            "team1": team1,
            "team2": team2
        }

        if "venue" in feature_columns:

            venues = sorted(
                df["venue"].dropna().astype(str).unique()
            )

            selected_venue = st.selectbox(
                "Select Match Venue",
                venues
            )

            input_values["venue"] = selected_venue

        if "toss_winner" in feature_columns:

            selected_toss = st.selectbox(
                "Select Toss Winner",
                [team1, team2]
            )

            input_values["toss_winner"] = selected_toss

        if "toss_decision" in feature_columns:

            selected_decision = st.selectbox(
                "Toss Decision",
                ["bat", "field"]
            )

            input_values["toss_decision"] = selected_decision

        st.caption(
            "For a meaningful prediction, enter match details "
            "that are known before the match starts."
        )

        if st.button("🔮 PREDICT MATCH WINNER"):

            if team1 == team2:

                st.error("Select two different teams.")

            else:

                prediction_input = pd.DataFrame(
                    [input_values],
                    columns=feature_columns
                )

                probabilities = model.predict_proba(
                    prediction_input
                )[0]

                classes = model.named_steps[
                    "classifier"
                ].classes_

                result = classes[np.argmax(probabilities)]
                confidence = np.max(probabilities) * 100

                st.success(
                    f"🏆 Predicted Winner: {result}"
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "Experimental prediction based on historical "
                    "match data. Not a guaranteed result."
                )

# ==================================================
# 4. IPL CHATBOT
# ==================================================

elif page == "💬 IPL Chatbot":

    st.subheader("💬 IPL AI Chatbot")

    st.caption(
        "Ask questions about IPL teams and statistics "
        "from your uploaded dataset."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask about IPL matches, teams or wins..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        q = question.lower()
        answer = ""

        if not valid_dataset or df.empty:

            answer = "Please upload a valid IPL dataset first."

        elif any(word in q for word in [
            "total matches", "how many matches",
            "number of matches", "match count"
        ]):

            answer = (
                f"The uploaded IPL dataset contains "
                f"{len(df)} valid match records."
            )

        elif any(word in q for word in [
            "top team", "most wins",
            "highest wins", "best team"
        ]):

            top = df["winner"].value_counts()

            if not top.empty:
                answer = (
                    f"The team with the most wins in this "
                    f"dataset is {top.index[0]}, "
                    f"with {top.iloc[0]} wins."
                )

        elif "season" in q and "season" in df.columns:

            seasons = sorted(
                df["season"].dropna().astype(str).unique()
            )

            answer = (
                "Available seasons in the dataset: "
                + ", ".join(seasons)
            )

        elif "venue" in q and "venue" in df.columns:

            venues = df["venue"].dropna().nunique()

            answer = (
                f"The dataset contains {venues} different venues."
            )

        elif "toss" in q and "toss_winner" in df.columns:

            toss_data = df.dropna(
                subset=["toss_winner"]
            )

            if len(toss_data) > 0:

                rate = (
                    toss_data["toss_winner"] ==
                    toss_data["winner"]
                ).mean() * 100

                answer = (
                    f"In {rate:.1f}% of matches with toss "
                    "information, the toss winner also won "
                    "the match."
                )

        else:

            matched_team = next(
                (
                    t for t in teams
                    if t.lower() in q
                ),
                None
            )

            if matched_team:

                played = (
                    (df["team1"] == matched_team) |
                    (df["team2"] == matched_team)
                ).sum()

                won = (
                    df["winner"] == matched_team
                ).sum()

                answer = (
                    f"📊 {matched_team}\n\n"
                    f"Matches played: {played}\n\n"
                    f"Matches won: {won}\n\n"
                    f"Matches not won: {played - won}"
                )

        if not answer:

            answer = (
                "I can answer questions about total matches, "
                "top winning teams, team records, venues, "
                "seasons and toss statistics. "
                "Try asking a more specific question."
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ==================================================
# 5. DATASET EXPLORER
# ==================================================

elif page == "📁 Dataset Explorer":

    st.subheader("📁 IPL Dataset Explorer")

    if not valid_dataset or df.empty:

        st.warning("Upload a valid IPL CSV first.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

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

        missing.columns = [
            "Column",
            "Missing Values"
        ]

        st.dataframe(
            missing,
            use_container_width=True,
            hide_index=True
        )

        st.write("### Dataset Columns")

        st.write(list(df.columns))

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Cleaned Dataset",
            data=csv_data,
            file_name="ipl_cleaned_data.csv",
            mime="text/csv"
        )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <h3>🏏 IPL AI ANALYST</h3>
    <p>Machine Learning Based IPL Analytics & Match Prediction</p>
    <p><b>Created by Anant OP</b></p>
    <p class="note">Educational Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)
