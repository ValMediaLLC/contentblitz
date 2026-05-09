import sys
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1":
    print(
        "Live API execution is disabled. "
        "Set CONTENTBLITZ_RUN_LIVE_TESTS=1 in .env to enable."
    )
    sys.exit(0)

from contentblitz.tools.search_web import search_web

def main():
    query = input("Enter search query: ").strip()

    result = search_web(
        query=query,
        max_results=5,
        provider="serp",
    )

    print("\nRESULT")
    print("------")
    print("Provider:", result.provider)
    print("Degraded:", result.degraded)
    print("Error:", result.error)

    print("\nResults:")
    for i, item in enumerate(result.results, start=1):
        print(f"\n{i}. {item.title}")
        print("URL:", item.url)
        print("Snippet:", item.snippet)


if __name__ == "__main__":
    main()