"""Carga de configuración y resolución de rutas.

Todas las rutas del config son relativas a la raíz del proyecto. Resolvemos esa raíz a
partir de la ubicación de este archivo, así da igual desde dónde se ejecute el código
(notebook, terminal, API): siempre apunta al mismo sitio.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "config.yaml") -> dict:
    with open(ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(relative: str) -> Path:
    """Convierte una ruta relativa del config en una ruta absoluta y real."""
    return ROOT / relative
