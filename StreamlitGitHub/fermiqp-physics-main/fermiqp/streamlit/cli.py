import streamlit.web.cli as stcli
import sys
from pathlib import Path

def run():
    sys.argv = ["streamlit", "run", str(Path(__file__).parent / "fermiqp-app.py")]
    sys.exit(stcli.main())