import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.app import app

# Vercel needs an ASGI app object to be exposed
# FastApi instance 'app' is sufficient
