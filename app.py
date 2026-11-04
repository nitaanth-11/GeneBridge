import os
import sys

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from genebridge_backend import app

# Standard entrypoint for Vercel and local testing
if __name__ == "__main__":
    app.run(debug=True, port=5000)
