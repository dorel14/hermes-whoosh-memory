"""Configuration commune des tests.

Les tests du fournisseur s'exécutent contre le VRAI contrat Hermes :
``agent/memory_provider.py`` du dépôt NousResearch/hermes-agent (module
100 % bibliothèque standard). La CI le récupère dans ``.hermes-contract/``.
En local : ``HERMES_CONTRACT_DIR=/chemin/contenant/agent pytest``.
Sans contrat disponible, les tests du fournisseur sont ignorés (skip).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_contract = os.environ.get("HERMES_CONTRACT_DIR") or str(ROOT / ".hermes-contract")
if Path(_contract, "agent", "memory_provider.py").is_file():
    sys.path.insert(0, _contract)
