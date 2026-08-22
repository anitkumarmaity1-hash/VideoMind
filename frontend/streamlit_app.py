"""
VideoMind v2 — Streamlit dashboard.
Layout: left = video player, right = question panel.
"""
import time
import streamlit as st
from services import api_client
from components.timestamp_player import render_video_player

st.set_page_config(page_title="VideoMind v2", layout="wide", page_icon="🎬")

if "video_id" not in st.session_state:
    st.session_state["video_id"] = None
if "seek_to" not in st.session_state:
    st.session_state["seek_to"] = 0

st.title("🎬 VideoMind v2 — Multimodal Video RAG")
st.caption(
    "Ask questions grounded in both what's said and what's shown in your video.")

with st.sidebar:
    st.header("Upload")
    tab_file, tab_url = st.tabs(["Upload file", "YouTube URL"])

    with tab_file:
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "mkv"])
        if uploaded and st.button("Process video", key="process_upload"):
            with st.spinner("Uploading..."):
                result = api_client.upload_video(
                    uploaded.getvalue(), uploaded.name)
                st.session_state["video_id"] = result["video_id"]
            st.success(f"Uploaded. Video ID: {result['video_id']}")

    with tab_url:
        url = st.text_input("YouTube URL")
        if url and st.button("Process URL", key="process_url"):
            with st.spinner("Fetching..."):
                result = api_client.upload_from_url(url)
                st.session_state["video_id"] = result["video_id"]
            st.success(f"Queued. Video ID: {result['video_id']}")

    st.divider()
    manual_id = st.text_input("Or load existing video_id")
    if manual_id:
        st.session_state["video_id"] = manual_id

video_id = st.session_state["video_id"]

if not video_id:
    st.info("Upload a video or paste a YouTube URL in the sidebar to get started.")
    st.stop()

# --- Processing status ---
status = api_client.get_status(video_id)
STAGE_ORDER = ["uploaded", "extracting_audio", "transcribing",
               "extracting_frames", "embedding", "indexing", "ready"]

if status["processing_status"] != "ready" and status["processing_status"] != "failed":
    st.warning(
        f"Processing status: **{status['processing_status']}** — this page will auto-refresh.")
    progress = STAGE_ORDER.index(status["processing_status"]) / (
        len(STAGE_ORDER) - 1) if status["processing_status"] in STAGE_ORDER else 0
    st.progress(progress)
    time.sleep(3)
    st.rerun()
elif status["processing_status"] == "failed":
    st.error(
        f"Processing failed: {status.get('error_message', 'unknown error')}")
    st.stop()

col_video, col_qa = st.columns([1, 1])

with col_video:
    st.subheader("Video")
    video_meta = api_client.get_video(video_id)
    # served via static mount in production
    video_url = f"{api_client.API_BASE_URL}/data/videos/{video_id}.mp4"
    st.caption(
        f"**{video_meta['filename']}** · {video_meta.get('duration', 0):.0f}s")
    render_video_player(video_url, st.session_state["seek_to"])

    with st.expander("Transcript segments"):
        segments = api_client.get_segments(video_id)
        for seg in segments:
            st.markdown(
                f"**[{seg['start_formatted']}–{seg['end_formatted']}]** {seg['transcript']}")

with col_qa:
    st.subheader("Ask a question")

    answer_mode = st.selectbox(
        "Answer mode", ["standard", "simple", "detailed", "technical"])
    question = st.text_input(
        "Your question", placeholder="What did the speaker say about AI?")

    suggested = [
        "Summarize this video",
        "Explain this video in simple English",
        "What is the main topic?",
        "What objects appear in the video?",
    ]
    st.caption("Suggested: " + " · ".join(suggested))

    if st.button("Ask") and question:
        with st.spinner("Retrieving evidence and generating answer..."):
            result = api_client.ask_question(video_id, question, answer_mode)
        st.markdown("### Answer")
        st.write(result["answer"])

    st.divider()
    st.subheader("Summary")
    summary_type = st.radio("Type", ["short", "detailed"], horizontal=True)
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            summary = api_client.get_summary(video_id, summary_type)
        if summary_type == "short":
            for b in summary["bullet_points"]:
                st.markdown(f"- {b}")
        else:
            for s in summary["sections"]:
                st.markdown(
                    f"**[{s['start_formatted']}–{s['end_formatted']}]** {s['summary']}")
