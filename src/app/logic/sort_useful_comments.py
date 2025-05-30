import os
import joblib
import numpy as np
from functools import lru_cache
from pydantic import BaseModel
from typing import List, Dict, Any

# Defines the output schema expected by the API
class CommentsOut(BaseModel):
    comments: List[Dict[str, Any]]  # ou adapte selon le contenu exact

# Loads the trained model once and caches it for reuse
@lru_cache
def _load_model():
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "useful-comments-sorter_model_xgb1.joblib")
    )
    return joblib.load(model_path)

# Filters and returns comments predicted as useful (label = 1)
def sort_useful_comments(*, comments: List[Dict[str, Any]]) -> CommentsOut:
    if not comments:
        return CommentsOut(comments=[])

    X = np.asarray([c["vector"] for c in comments], dtype=float)
    preds = _load_model().predict(X)

    useful_comments = [c for c, y in zip(comments, preds) if y == 1]

    return CommentsOut(comments=useful_comments)