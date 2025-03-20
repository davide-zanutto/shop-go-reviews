import json
import streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

llm_client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)

model = "gpt-4o-mini"

def load_data(json_file):
    """Load the JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)

# Extract keywords and their reviews
def get_keywords_and_reviews(data):
    """
    Extract all keywords and their associated reviews and subtopics.
    Returns:
        - keyword_map: Dictionary mapping display keywords to JSON keys.
        - reviews_by_keyword: Dictionary mapping JSON keys to their subtopics and reviews.
    """
    keyword_map = {}
    reviews_by_keyword = {}

    for json_key, keyword_data in data.items():
        display_keyword = keyword_data["Keyword"] 
        keyword_map[display_keyword] = json_key

        # Extract subtopics and reviews
        subtopics = keyword_data["Subtopics"]
        reviews_by_keyword[json_key] = {
            "Reviews": [review for subtopic in subtopics.values() for review in subtopic["Reviews"]],
            "Subtopics": {
                subtopic_data["Subtopic_keyword"]: subtopic_data["Reviews"]
                for subtopic_data in subtopics.values()
            },
        }

    return keyword_map, reviews_by_keyword

# Streamlit app
def main():
    st.title("Shop & Go reviews")
    
    json_file = "topics.json"
    data = load_data(json_file)

    keyword_map, reviews_by_keyword = get_keywords_and_reviews(data)

    if "Summary" not in st.session_state:
        st.session_state.short_summary = None
        st.session_state.long_summary = None 

    # Sidebar for filters
    st.sidebar.title("Actions")
    st.sidebar.write(f"### Filter reviews by topic")

    selected_keyword = st.sidebar.selectbox("Select a Keyword", options=["All Keywords"] + list(keyword_map.keys()))

    # Initialize `all_reviews` based on the selected filters
    all_reviews = []

    if selected_keyword == "All Keywords":
        all_reviews = [review for json_key in keyword_map.values() for review in reviews_by_keyword[json_key]["Reviews"]]
    else:
        json_key = keyword_map[selected_keyword]

        subtopics = list(reviews_by_keyword[json_key]["Subtopics"].keys())
        selected_subtopic = st.sidebar.selectbox("Select a Subtopic", options=["All Subtopics"] + subtopics)

        if selected_subtopic == "All Subtopics":
            all_reviews = reviews_by_keyword[json_key]["Reviews"]
        else:
            all_reviews = reviews_by_keyword[json_key]["Subtopics"][selected_subtopic]

    st.sidebar.write(f"### Generate a summary")
    # Sidebar: Display Summary
    if selected_keyword != "All Keywords":
        json_key = keyword_map[selected_keyword]

        if st.sidebar.button("Show Short Summary"):
            st.session_state.short_summary = data[json_key].get("Short summary", "No short summary available.")

        if st.sidebar.button("Show Detailed Summary"):
            st.session_state.long_summary = data[json_key].get("Long summary", "No detailed summary available.")

    if st.session_state.short_summary:
        formatted_summary = st.session_state.short_summary.replace("\n", "<br>")
        st.markdown(f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:8px;">
            <h3>Generated short summary</h3>
            {formatted_summary}
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.long_summary:
        formatted_summary = st.session_state.long_summary.replace("\n", "<br>")
        st.markdown(f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:8px;">
            <h3>Generated detailed summary</h3>
            {formatted_summary}
        </div>
        """, unsafe_allow_html=True)


    # Display the reviews based on the filters
    if all_reviews:
        st.write("### Reviews")
        for review in all_reviews:
            st.write(f"- {review}")
    else:
        st.write("### No reviews available for the selected filters.")

if __name__ == "__main__":
    main()