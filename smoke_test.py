import os

from app import get_transcript, scrape_goodreads

GOODREADS_URL = "https://www.goodreads.com/list/show/1.Best_Books_Ever"
VIDEO_IDS = [
    "dQw4w9WgXcQ",
    "3JZ_D3ELwOQ",
    "kJQP7kiw5Fk",
    "9bZkp7q19f0",
    "fJ9rUzIMcZQ",
    "hTWKbfoikeg",
    "L_jWHffIx5E",
    "YQHsXMglC9A",
    "Zi_XLOBDo_Y",
    "60ItHLz5WEA",
    "M7FIvfx5J10",
    "uelHwf8o7_U",
]


def ensure_env():
    required = ["BRIGHTDATA_API_KEY", "BRIGHTDATA_ZONE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing vars for smoke test: {', '.join(missing)}")


def test_goodreads_scrape():
    books = scrape_goodreads(GOODREADS_URL)
    assert len(books) >= 5, f"Expected >=5 books, got {len(books)}"
    print(f"Goodreads scrape OK: {len(books)} books")


def test_transcripts():
    success = 0
    attempted = 0
    for vid in VIDEO_IDS:
        attempted += 1
        text, source = get_transcript(vid)
        if text:
            success += 1
            print(f"Transcript OK ({success}/3): {vid} via {source}, len={len(text)}")
            if success >= 3:
                break
        else:
            if source in {"no_captions", "unavailable"}:
                print(f"Skip {vid}: genuinely unavailable ({source})")
            else:
                print(f"Failed {vid}: {source}")

    assert success >= 3, f"Need 3 transcript successes, got {success} after {attempted} attempts"


if __name__ == "__main__":
    ensure_env()
    test_goodreads_scrape()
    test_transcripts()
    print("Smoke tests passed")
