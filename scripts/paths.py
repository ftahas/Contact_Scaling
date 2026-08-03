"""Where the scripts read and write.

Set CONTACT_DATA or CONTACT_FIGS to put the output somewhere else.
"""
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.environ.get('CONTACT_DATA', os.path.join(_root, 'data'))
FIGS = os.environ.get('CONTACT_FIGS', os.path.join(_root, 'figures'))

os.makedirs(DATA, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
