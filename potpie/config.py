"""Runtime configuration for Potpie."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class PotpieConfig:
    """Configuration for Potpie runtime.

    Attributes:
        project_path: Root path of the project to analyze.
        model: LLM model identifier (LiteLLM format).
        db_path: Path to SQLite database. Auto-generated in .potpie/ if None.
        graph_path: Path to graph JSON file. Auto-generated in .potpie/ if None.
        user_id: User identifier. Defaults to "local-user" for local mode.
    """

    project_path: str = "."
    model: str = field(default_factory=lambda: os.getenv("POTPIE_MODEL", "gpt-4o"))
    db_path: Optional[str] = None
    graph_path: Optional[str] = None
    user_id: str = "local-user"

    def __post_init__(self):
        project = Path(self.project_path).resolve()
        potpie_dir = project / ".potpie"

        if self.db_path is None:
            self.db_path = str(potpie_dir / "potpie.db")

        if self.graph_path is None:
            self.graph_path = str(potpie_dir / "graph.json")

    def ensure_directories(self) -> None:
        """Create .potpie directory if it doesn't exist."""
        if self.db_path:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if self.graph_path:
            Path(self.graph_path).parent.mkdir(parents=True, exist_ok=True)
