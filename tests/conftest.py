import os
import sys

# Make backend/app importable as `app.*` when running pytest from repo root
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))
