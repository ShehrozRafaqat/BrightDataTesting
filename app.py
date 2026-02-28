import base64
import html
import json
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY", "")
BRIGHTDATA_ZONE = os.getenv("BRIGHTDATA_ZONE", "")
BRIGHTDATA_API_URL = os.getenv("BRIGHTDATA_API_URL", "https://api.brightdata.com/request")

LAST_BOOKS: List[Dict[str, str]] = []
LAST_TRANSCRIPT: str = ""


def _decode_brightdata_response(resp: requests.Response) -> bytes:
    content_type = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return resp.content

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return resp.content

    if isinstance(data, dict):
        if isinstance(data.get("body"), str):
            body = data["body"]
            if data.get("body_encoding") == "base64":
                return base64.b64decode(body)
            return body.encode("utf-8", errors="ignore")
        if isinstance(data.get("result"), str):
            return data["result"].encode("utf-8", errors="ignore")
        if isinstance(data.get("data"), str):
            try:
                return base64.b64decode(data["data"])
            except Exception:
                return data["data"].encode("utf-8", errors="ignore")
    return resp.content


def brightdata_fetch(url: str, timeout: int = 60) -> bytes:
    if not BRIGHTDATA_API_KEY or not BRIGHTDATA_ZONE:
        raise RuntimeError("Missing Bright Data credentials in environment.")

    headers = {
        "Authorization": f"Bearer {BRIGHTDATA_API_KEY}",
        "Accept": "*/*",
    }
    params = {
        "url": url,
        "zone": BRIGHTDATA_ZONE,
    }

    resp = requests.get(BRIGHTDATA_API_URL, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return _decode_brightdata_response(resp)


def scrape_goodreads(url: str) -> List[Dict[str, str]]:
    html_doc = brightdata_fetch(url).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_doc, "lxml")
    books: List[Dict[str, str]] = []

    for row in soup.select("tr"):
        title_el = row.select_one("a.bookTitle span") or row.select_one("a.bookTitle")
        author_el = row.select_one("a.authorName span") or row.select_one("a.authorName")
        rating_el = row.select_one("span.minirating")

        title = title_el.get_text(strip=True) if title_el else ""
        author = author_el.get_text(strip=True) if author_el else ""
        rating = rating_el.get_text(" ", strip=True) if rating_el else ""

        if title:
            books.append(
                {
                    "Book Title": title,
                    "Author Name": author or "N/A",
                    "Star Rating": rating or "N/A",
                }
            )

    seen = set()
    deduped: List[Dict[str, str]] = []
    for book in books:
        key = (book["Book Title"], book["Author Name"])
        if key not in seen:
            seen.add(key)
            deduped.append(book)
    return deduped


def ask_groq(question: str, context: str) -> str:
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY. Add it to the environment/secrets."

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer using only provided context when possible. If context lacks details, say so clearly.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content or "No response generated."


def _youtube_api_transcript(video_id: str) -> Tuple[Optional[str], str]:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(seg.get("text", "") for seg in segments).strip()
        if text:
            return text, "ok"
        return None, "empty"
    except (NoTranscriptFound, TranscriptsDisabled):
        return None, "no_captions"
    except VideoUnavailable:
        return None, "unavailable"
    except Exception:
        return None, "blocked"


def _extract_caption_tracks(player_response: Dict) -> List[Dict]:
    try:
        return (
            player_response.get("captions", {})
            .get("playerCaptionsTracklistRenderer", {})
            .get("captionTracks", [])
        )
    except Exception:
        return []


def _pick_caption_track(tracks: List[Dict]) -> Optional[Dict]:
    preferred = ["en", "ur", "en-US", "en-GB"]
    ranked: List[Tuple[int, int, Dict]] = []

    for track in tracks:
        lang = (track.get("languageCode") or "").strip()
        if lang not in preferred:
            continue
        lang_rank = preferred.index(lang)
        is_asr = 1 if track.get("kind") == "asr" else 0
        ranked.append((lang_rank, is_asr, track))

    if not ranked:
        return tracks[0] if tracks else None
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked[0][2]


def _brightdata_transcript(video_id: str) -> Tuple[Optional[str], str]:
    info_url = (
        "https://www.youtube.com/get_video_info?"
        + urlencode(
            {
                "video_id": video_id,
                "el": "detailpage",
                "hl": "en",
                "html5": "1",
            }
        )
    )

    raw = brightdata_fetch(info_url).decode("utf-8", errors="ignore")
    qs = parse_qs(raw)
    player_response_raw = qs.get("player_response", [None])[0]
    if not player_response_raw:
        return None, "no_player_response"

    try:
        player_response = json.loads(player_response_raw)
    except json.JSONDecodeError:
        return None, "bad_player_response"

    tracks = _extract_caption_tracks(player_response)
    if not tracks:
        return None, "no_captions"

    track = _pick_caption_track(tracks)
    if not track or not track.get("baseUrl"):
        return None, "no_captions"

    xml_bytes = brightdata_fetch(track["baseUrl"])
    xml_text = xml_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(xml_text, "xml")
    chunks = []
    for node in soup.find_all("text"):
        content = node.get_text(" ", strip=True)
        if content:
            chunks.append(html.unescape(content))
    transcript = " ".join(chunks).strip()
    if not transcript:
        return None, "empty"
    return transcript, "ok"


def get_transcript(video_id: str) -> Tuple[Optional[str], str]:
    text, status = _youtube_api_transcript(video_id)
    if text:
        return text, "youtube_transcript_api"

    if status in {"no_captions", "unavailable"}:
        return None, status

    fallback_text, fallback_status = _brightdata_transcript(video_id)
    if fallback_text:
        return fallback_text, "brightdata"
    return None, fallback_status


def scrape_books_ui(url: str):
    global LAST_BOOKS
    try:
        books = scrape_goodreads(url)
        LAST_BOOKS = books
        msg = f"Extracted {len(books)} books from Goodreads page."
        return books, json.dumps(books[:10], indent=2), msg
    except Exception as exc:
        LAST_BOOKS = []
        return [], "[]", f"Scrape failed: {exc}"


def books_chat_fn(message, history):
    if not LAST_BOOKS:
        return "Please scrape a Goodreads URL first."
    context = json.dumps(LAST_BOOKS[:100], ensure_ascii=False)
    return ask_groq(message, context)


def load_transcript_ui(video_id: str):
    global LAST_TRANSCRIPT
    vid = (video_id or "").strip()
    if not vid:
        LAST_TRANSCRIPT = ""
        return "", "Please provide a YouTube VIDEO ID (not full URL)."

    try:
        transcript, source = get_transcript(vid)
        if transcript:
            LAST_TRANSCRIPT = transcript
            return transcript, f"Transcript loaded via {source}."
        LAST_TRANSCRIPT = ""
        return "", "Transcript unavailable for this video (captions missing/disabled)."
    except Exception as exc:
        LAST_TRANSCRIPT = ""
        return "", f"Failed to load transcript: {exc}"


def transcript_chat_fn(message, history):
    if not LAST_TRANSCRIPT:
        return "Load a transcript first in Tab 2."
    return ask_groq(message, LAST_TRANSCRIPT[:15000])


def build_app():
    with gr.Blocks(title="Programming Assignment Ver2") as demo:
        gr.Markdown("# Programming Assignment Ver2")

        with gr.Tabs():
            with gr.TabItem("Goodreads Scraper + Q&A"):
                gr.Markdown("Scrape bot-protected page with Bright Data Web Unlocker, then ask questions.")
                url_input = gr.Textbox(
                    value="https://www.goodreads.com/list/show/1.Best_Books_Ever",
                    label="Target URL",
                )
                scrape_btn = gr.Button("Scrape with Bright Data")
                status_box = gr.Textbox(label="Status", interactive=False)
                books_df = gr.Dataframe(headers=["Book Title", "Author Name", "Star Rating"], label="Extracted Books")
                books_json = gr.Code(label="Books JSON Preview", language="json")

                scrape_btn.click(fn=scrape_books_ui, inputs=[url_input], outputs=[books_df, books_json, status_box])

                gr.ChatInterface(
                    fn=books_chat_fn,
                    title="Ask Questions About Scraped Books",
                    description="Uses Groq LLM on top of scraped Goodreads content.",
                )

            with gr.TabItem("YouTube Transcript Q&A"):
                gr.Markdown("Enter YouTube VIDEO ID, fetch transcript, then ask questions.")
                vid_input = gr.Textbox(label="YouTube VIDEO ID", placeholder="e.g., dQw4w9WgXcQ")
                load_btn = gr.Button("Load Transcript")
                transcript_status = gr.Textbox(label="Status", interactive=False)
                transcript_box = gr.Textbox(label="Transcript", lines=12)

                load_btn.click(fn=load_transcript_ui, inputs=[vid_input], outputs=[transcript_box, transcript_status])

                gr.ChatInterface(
                    fn=transcript_chat_fn,
                    title="Ask Questions About Transcript",
                    description="Tries youtube-transcript-api first, then Bright Data fallback.",
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
