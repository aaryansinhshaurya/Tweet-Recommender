
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    """
    Loads tweet embeddings and tweet texts from CSV files.
    Converts tweet IDs to integers and parses the embeddings.
    """
    tweet_embeddings_df = pd.read_csv("embeddings.csv")
    tweet_texts_df = pd.read_csv("datasetFinal.csv")
    
    # Ensure tweet IDs are integers
    tweet_embeddings_df["tweet_id"] = tweet_embeddings_df["tweet_id"].astype(int)
    tweet_texts_df["tweet_id"] = tweet_texts_df["tweet_id"].astype(int)

    def parse_embedding(x):
        # Remove brackets and newlines, then split by space
        cleaned = x.replace("[", "").replace("]", "").replace("\n", " ")
        arr = np.fromstring(cleaned, dtype=float, sep=" ")
        return arr

    tweet_embeddings_df["sentence_embedding"] = tweet_embeddings_df["sentence_embedding"].apply(parse_embedding)
    tweet_embeddings_df.dropna(subset=["sentence_embedding"], inplace=True)

    return tweet_embeddings_df, tweet_texts_df

# ------------------- INITIALIZE SESSION STATE -------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

if "liked_tweets" not in st.session_state:
    st.session_state["liked_tweets"] = []

if "posted_tweets" not in st.session_state:
    st.session_state["posted_tweets"] = []

# Load dataframes into session state if not already present
if "tweet_embeddings_df" not in st.session_state or "tweet_texts_df" not in st.session_state:
    embeddings_df, texts_df = load_data()
    st.session_state["tweet_embeddings_df"] = embeddings_df
    st.session_state["tweet_texts_df"] = texts_df
else:
    embeddings_df = st.session_state["tweet_embeddings_df"]
    texts_df = st.session_state["tweet_texts_df"]

# Initialize SentenceTransformer model only once
if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- AUTH / LOGIN PAGE -------------------
def login_page():
    st.title("Tweet Recommender - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:
            st.session_state["user"] = username
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Please enter valid credentials.")

if not st.session_state["user"]:
    login_page()
    st.stop()

# ------------------- DEBUG: SHOW SESSION STATE -------------------
# st.sidebar.markdown("### Debug: Session State")
# st.sidebar.write({
#     "Liked Tweets": st.session_state["liked_tweets"],
#     "Posted Tweets": st.session_state["posted_tweets"]
# })

# ------------------- LIKE CALLBACK FUNCTION -------------------
def like_tweet(tid):
    tid = int(tid)  # Ensure type consistency
    if tid not in st.session_state["liked_tweets"]:
        st.session_state["liked_tweets"].append(tid)
        st.success(f"Tweet {tid} Liked!")
    else:
        st.info("Already liked!")
    # No need to call st.rerun() here; the state update persists

# ------------------- RECOMMENDATION FUNCTION -------------------
def get_recommendations():
    """
    Averages embeddings of liked and posted tweets,
    computes cosine similarity,
    and returns top recommended tweet IDs with at least 5 words.
    """
    user_liked_ids = st.session_state["liked_tweets"]
    user_posted_ids = st.session_state["posted_tweets"]

    liked_embeddings = embeddings_df[embeddings_df["tweet_id"].isin(user_liked_ids)]["sentence_embedding"].tolist()
    posted_embeddings = embeddings_df[embeddings_df["tweet_id"].isin(user_posted_ids)]["sentence_embedding"].tolist()

    if not liked_embeddings and not posted_embeddings:
        st.write("No liked or posted tweets found.")
        return []

    combined = liked_embeddings + posted_embeddings
    avg_embedding = np.mean(combined, axis=0).reshape(1, -1)

    all_tweet_embeddings = np.vstack(embeddings_df["sentence_embedding"])
    similarities = cosine_similarity(avg_embedding, all_tweet_embeddings)[0]

    # Work on a copy to avoid modifying the original DataFrame
    temp_df = embeddings_df.copy()
    temp_df["similarity"] = similarities

    # Merge with text DataFrame to filter based on word count
    temp_df = temp_df.merge(texts_df, on="tweet_id")

    # Count words in the tweet text
    temp_df["word_count"] = temp_df["original_text"].apply(lambda x: len(str(x).split()))

    # Filter tweets with at least 5 words
    recommended = temp_df[temp_df["word_count"] >= 5].sort_values(by="similarity", ascending=False)

    exclude_ids = user_liked_ids + user_posted_ids
    recommended = recommended[~recommended["tweet_id"].isin(exclude_ids)]

    if recommended.empty:
        return temp_df[temp_df["word_count"] >= 5]["tweet_id"].sample(min(5, len(temp_df))).tolist()

    return recommended.head(10)["tweet_id"].tolist()

# ------------------- EXPLORE PAGE -------------------
def explore_page():
    st.title("Explore Tweets")

    if embeddings_df.empty:
        st.warning("No tweets available! Please check your dataset.")
        return

    sample_size = min(10, len(embeddings_df))
    tweet_ids = embeddings_df["tweet_id"].sample(sample_size).tolist()

    for tid in tweet_ids:
        text = texts_df.loc[texts_df["tweet_id"] == tid, "original_text"].values[0]
        st.write(text)

        # Use on_click callback for the Like button
        st.button(f"Like ❤️", key=f"like_{tid}", on_click=like_tweet, args=(tid,))
        st.markdown("---")

    # st.markdown("---")
    
    st.title("Post Tweets")
    st.subheader("Write a Tweet")
    new_tweet = st.text_area("What's on your mind?", key="tweet_input", label_visibility="collapsed")
    if st.button("Post Tweet"):
        if new_tweet.strip():
            new_tweet_id = texts_df["tweet_id"].max() + 1

            st.session_state["posted_tweets"].append(new_tweet_id)

            new_text_row = pd.DataFrame({
                "tweet_id": [new_tweet_id],
                "original_text": [new_tweet]
            })
            st.session_state["tweet_texts_df"] = pd.concat([texts_df, new_text_row], ignore_index=True)

            model = st.session_state["embedding_model"]
            new_embedding = model.encode([new_tweet])[0]

            new_embedding_row = pd.DataFrame({
                "tweet_id": [new_tweet_id],
                "sentence_embedding": [new_embedding]
            })
            st.session_state["tweet_embeddings_df"] = pd.concat([embeddings_df, new_embedding_row], ignore_index=True)

            st.success("Tweet Posted & Embedding Added!")
            st.rerun()

# ------------------- FOR YOU PAGE -------------------
def for_you_page():
    st.title("For You")

    recommended_ids = get_recommendations()
    if not recommended_ids:
        st.write("No recommendations yet. Like or post tweets to get suggestions!")
        return

    for tid in recommended_ids:
        text = texts_df.loc[texts_df["tweet_id"] == tid, "original_text"].values[0]
        st.write(text)

        st.button(f"Like ❤️", key=f"rec_like_{tid}", on_click=like_tweet, args=(tid,))

# ------------------- MAIN APP -------------------
st.sidebar.title(f"Welcome, {st.session_state['user']}")
page = st.sidebar.radio("Navigate", ["Explore", "For You", "Logout"])

if page == "Explore":
    explore_page()
elif page == "For You":
    for_you_page()
elif page == "Logout":
    st.session_state["user"] = None
    st.session_state["liked_tweets"] = []
    st.session_state["posted_tweets"] = []
    st.rerun()

# # Optional full session state debug info
# st.sidebar.markdown("### Full Session State")
# st.sidebar.write(st.session_state)