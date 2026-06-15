import sys
import os

# Ensure the parent directory is in the path so we can import genebridge_backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from genebridge_backend import app
