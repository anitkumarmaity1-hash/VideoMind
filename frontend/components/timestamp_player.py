"""
Reusable component: renders clickable [MM:SS] timestamp buttons that seek
the video player. Streamlit doesn't allow direct JS->Python video seeking
without a custom component, so we implement seeking via a query-param
driven re-render of an HTML5 <video> tag with a `#t=` fragment, which all
major browsers honor for local/served video files.
"""
import streamlit as st


def render_video_player(video_url: str, seek_seconds: float = 0):
    st.markdown(
        f"""
        <video id="videomind-player" width="100%" controls autoplay="false">
            <source src="{video_url}#t={seek_seconds}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_timestamps(evidence_items, key_prefix: str = "ev"):
    """
    Renders a row of [MM:SS] buttons for each evidence item. Clicking one
    sets st.session_state['seek_to'] which the caller uses to re-render
    the video player at that timestamp.
    """
    cols = st.columns(len(evidence_items)) if evidence_items else []
    for i, item in enumerate(evidence_items):
        label = f"[{item['start_formatted']}]"
        if cols[i].button(label, key=f"{key_prefix}_{i}"):
            st.session_state["seek_to"] = item["start_time"]
            st.rerun()
