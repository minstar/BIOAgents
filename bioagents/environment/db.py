"""Base database class for BIOAgents domains."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


class DB(BaseModel):
    """Domain database base class.
    
    All domain databases (patient records, drug databases, etc.) inherit from this.
    Provides JSON serialization, loading, and hashing.
    """

    @classmethod
    def load(cls, path: str) -> "DB":
        """Load the database from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def dump(self, path: str) -> None:
        """Dump the database to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False, default=str)

    def get_hash(self) -> str:
        """Get a deterministic hash of the database state."""
        data_str = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON schema of the database."""
        return self.model_json_schema()

    def update(self, update_data: dict[str, Any]) -> None:
        """Update database fields with new data."""
        for key, value in update_data.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, dict) and isinstance(value, dict):
                    current.update(value)
                else:
                    setattr(self, key, value)
