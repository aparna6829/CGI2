import streamlit as st
from youtubesearchpython import VideosSearch

def search_youtube_videos(user_input):
    search = VideosSearch(user_input, limit=5)
    ytlinks = ""
    search_results = search.result().get('result', [])

    if not search_results:
        ytlinks = "No links available on YouTube related to this."

    for i, result in enumerate(search_results):
        title = result.get('title', 'No title')
        video_id = result.get('id', '')
        if video_id:
            ytlinks += f"{i + 1}. Title: {title} Link: https://www.youtube.com/watch?v={video_id}\n"

    if not ytlinks:
        ytlinks = "No links available on YouTube related to this."

    return ytlinks

def main():
    st.title("YouTube Video Search")

    user_input_diagnosis = st.text_input("Enter diagnosis keyword:")
    
    if st.button("Search"):
        ytlinks = search_youtube_videos(user_input_diagnosis)
        st.markdown(ytlinks)

if __name__ == "__main__":
    main()
