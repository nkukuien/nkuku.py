import streamlit as st
import openai
import requests
import cv2
from clarifai.rest import Image as ClarifaiImage
from clarifai.rest import ClarifaiApp
from bs4 import BeautifulSoup
import random

# Set your OpenAI GPT-3 API key and Clarifai API key here
gpt3_api_key = "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"
clarifai_api_key = "YOUR_CLARIFAI_API_KEY"

openai.api_key = gpt3_api_key
clarifai_app = ClarifaiApp(api_key=clarifai_api_key)

# Function to generate a summary using GPT-3
def generate_summary(description, template="The football goal {goal_description}"):
    prompt = template.format(goal_description=description)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to analyze an image using Clarifai
def analyze_image(image_url):
    model = clarifai_app.public_models.general_model
    image = ClarifaiImage(url=image_url)
    response = model.predict([image])
    concepts = response['outputs'][0]['data']['concepts']
    return [concept['name'] for concept in concepts]

# Function to analyze a video (improved goal detection logic using OpenCV)
def analyze_video(video_url):
    response = requests.get(video_url)
    video_path = "temp_video.mp4"
    with open(video_path, 'wb') as f:
        f.write(response.content)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    goals_detected = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Improved goal detection logic using OpenCV (replace with your logic)
        if meets_goal_condition(frame):
            goals_detected.append((i, frame))  # Store frame number and frame where goal detected

    cap.release()
    cv2.destroyAllWindows()

    return goals_detected

# Placeholder function for goal detection condition (replace with actual detection logic)
def meets_goal_condition(frame):
    return True  # Placeholder condition - always returns True

# Streamlit app layout
def main():
    st.title("Football Video Goal App")
    st.markdown("---")

    query = st.text_input("Search for goals", help="Enter a description to search for specific goals.")
    league = st.selectbox("Select a league", ["NBC Tanzania", "EPL", "La Liga", "Serie A", "Other Leagues", "Popular"])

    input_type = st.radio("Select Input Type", ("Image", "Video"), help="Choose between image or video analysis.")

    if input_type == "Image":
        image_url = st.text_input("Enter Image URL", help="Provide a direct link to an image.")
        if st.button("Analyze Image"):
            if image_url:
                image_tags = analyze_image(image_url)
                st.write("Image Tags:", image_tags)
            else:
                st.warning("Please provide an image URL.")
    else:
        video_url = st.text_input("Enter Video URL", help="Provide a direct link to a video.")
        if st.button("Analyze Video"):
            if video_url:
                detected_frames = analyze_video(video_url)
                st.write(f"Detected {len(detected_frames)} Frames Containing Goals:")
                for i, frame in detected_frames:
                    st.image(frame, caption=f"Frame {i}", use_column_width=True)
            else:
                st.warning("Please provide a video URL.")

    if st.button("Search"):
        if league == "Popular":
            goal_clips = fetch_random_goals()
        else:
            goal_clips = fetch_goal_clips(query, league)
        summarized_clips = []
        for clip in goal_clips:
            summary = generate_summary_gpt3(clip['description'], clip['url'])
            summarized_clips.append({'clip': clip, 'summary': summary})

        for clip in summarized_clips:
            st.subheader(clip['clip']['description'])
            st.write(clip['summary'])

# Function to fetch random goals from the fetched clips
def fetch_random_goals():
    goal_clips = fetch_goal_clips("", "")  # Fetch all goal clips
    random.shuffle(goal_clips)
    return random.sample(goal_clips, min(len(goal_clips), 10))  # Select a random subset of clips

# Function to fetch goal clips from YouTube playlists and channels
def fetch_goal_clips(query, league):
    goal_clips = []

    # Scraping YouTube playlists and channels for goal clips
    urls = [
        "https://www.youtube.com/playlist?list=PLQ_voP4Q3cffZYz6sVkSigiLfAZI_5vba",
        "https://t.me/s/skysports_goals",
        "https://www.youtube.com/@LaLiga/shorts",
        "https://www.youtube.com/@bundesliga/shorts",
        "https://www.youtube.com/@bundesliga/videos",
        "https://www.youtube.com/@FootballLiveGameplay/videos",
        "https://www.youtube.com/@FootballTVuefa/shorts"
    ]

    for url in urls:
        response = requests.get(url)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            videos = soup.find_all('a', href=True)
            for video in videos:
                if 'watch?v=' in video['href']:
                    goal_clips.append({'description': video.text.strip(), 'url': f"https://www.youtube.com{video['href']}"})

    return goal_clips

# Function to generate a summary using GPT-3 (modified to fit Streamlit)
def generate_summary_gpt3(description, video_url):
    prompt = f"Summarize the football goal: {description}. Provide details about the video: {video_url}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    main()
