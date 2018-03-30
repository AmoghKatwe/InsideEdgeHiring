"""Import python_hiring_test."""
import os

# set up some basic path names for easier compatibility
ROOT = os.path.dirname(__file__)

DATA = os.path.join(ROOT, 'data')

RAW = os.path.join(DATA, 'raw')
REFERENCE = os.path.join(DATA, 'reference')
PROCESSED = os.path.join(DATA, 'processed')
