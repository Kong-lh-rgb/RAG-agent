import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline import RAGPipeline
import json

def test():
    pipeline = RAGPipeline()
    try:
        stream = pipeline.query_and_generate_stream(
            text="Hello",
            top_k=3,
            doc_ids=[],
            session_id="test_sess"
        )
        for chunk in stream:
            print(chunk.strip())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
