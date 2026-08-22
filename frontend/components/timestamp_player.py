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


def render_evidence_timestamps(evidence_items, key_prefix: str = "ev", per_row: int = 6):
    """
    Renders [MM:SS] buttons for each evidence item, wrapped into rows of
    `per_row` so columns never get squeezed too narrow to render the label
    (a single st.columns(len(evidence_items)) row breaks down once there
    are more than ~6-8 items, wrapping each label character-by-character).
    Clicking a button sets st.session_state['seek_to'] which the caller
    uses to re-render the video player at that timestamp.
    """
    if not evidence_items:
        return
    for row_start in range(0, len(evidence_items), per_row):
        row_items = evidence_items[row_start:row_start + per_row]
        cols = st.columns(per_row)
        for i, item in enumerate(row_items):
            label = f"[{item['start_formatted']}]"
            idx = row_start + i
            if cols[i].button(label, key=f"{key_prefix}_{idx}"):
                st.session_state["seek_to"] = item["start_time"]
                st.rerun()
