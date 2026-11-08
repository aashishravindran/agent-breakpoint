"""Serve the scripted fake model: python -m examples.toy_bank.serve_llm [port]"""

import sys

import uvicorn

from examples.toy_bank.fake_llm import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765,
                log_level="warning")
