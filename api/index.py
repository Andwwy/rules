"""Vercel Python entrypoint.

Vercel's @vercel/python runtime serves the WSGI `app` defined/imported here.
We import the Flask app from annotate.py at the project root. All requests are
rewritten to this function by vercel.json.
"""
import os
import sys

# make the project root (where annotate.py lives) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotate import app  # noqa: E402  (Flask WSGI app)
