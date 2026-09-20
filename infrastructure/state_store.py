import json
from pathlib import Path

class JsonStateStore:
    """Persistence boundary for live Event Opportunity state."""
    def __init__(self, path): self.path = Path(path)
    def _load(self):
        if not self.path.exists(): return {}
        return json.loads(self.path.read_text())
    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    def get_opportunity(self, opportunity_id): return self._load().get(opportunity_id)
    def list_opportunities(self): return list(self._load().values())
    def save_opportunity(self, state):
        data=self._load(); data[state["id"]]=state; self._save(data)
