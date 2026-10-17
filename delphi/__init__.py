"""Delphi: a library for statistical estimation under truncation."""

import logging

NAME = "delphi"
__version__ = "1.0.0"

# A library must not install handlers of its own; the application decides
# where records go. Callers opt in with logging.basicConfig(level=...).
logging.getLogger(NAME).addHandler(logging.NullHandler())
