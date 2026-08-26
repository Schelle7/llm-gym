import json

from llm_gym.config import ROOT
from llm_gym.server.app import app

OUTPUT = ROOT / "frontend" / "openapi.json"

OUTPUT.write_text(json.dumps(app.openapi(), indent=2) + "\n")