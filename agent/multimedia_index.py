from __future__ import annotations

"""
Multimedia index and helpers.

This module connects movie / person entities to image IDs that can be
displayed in the Speakeasy frontend.

According to the TA instructions, the frontend expects messages like:

    image:0000/1HJA8MiuTYGf40u7x7Bgw3Yv7py

We therefore only need to return strings of the form
"<subfolder>/<image_id>" without the ".jpg" suffix. The GraphExecutor
already normalises image IDs this way when it parses images.json.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .graph_executor import GraphExecutor, get_global_graph_executor


@dataclass
class MultimediaItem:
    """
    Simple container representing multimedia for one entity.
    """

    uri: str
    label: Optional[str]
    image_ids: List[str]


class MultimediaIndex:
    """
    Thin wrapper around GraphExecutor for multimedia access.

    It knows:
      - how to retrieve image ids for a movie or person URI
      - how to pick a small subset of "best" images
      - how to format them as Speakeasy image tokens
    """

    def __init__(self, graph: Optional[GraphExecutor] = None) -> None:
        self.graph: GraphExecutor = graph or get_global_graph_executor()

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def get_image_ids_for_uri(self, uri: str) -> List[str]:
        """
        Return a list of image ids "0000/xxxx" for a given entity URI.

        If there are no images known, return an empty list.
        """
        idx = self.graph.get_index_for_uri(uri)
        if idx is None:
            return []
        return list(self.graph.get_images(idx))

    def get_best_image_id_for_uri(self, uri: str) -> Optional[str]:
        """
        Return a single image id for the entity, choosing the first one.

        This keeps the UI simple. If you want multiple images, call
        `get_image_ids_for_uri` directly.
        """
        images = self.get_image_ids_for_uri(uri)
        if not images:
            return None
        return images[0]

    def get_multimedia_item(self, uri: str) -> Optional[MultimediaItem]:
        """
        Package uri + label + image_ids into one object.
        """
        idx = self.graph.get_index_for_uri(uri)
        if idx is None:
            return None
        label = self.graph.get_label(idx)
        image_ids = self.graph.get_images(idx)
        if not image_ids:
            return None
        return MultimediaItem(uri=uri, label=label, image_ids=list(image_ids))

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def to_speakeasy_token(image_id: str) -> str:
        """
        Convert a normalised image id (subfolder/ID) into the token that
        the Speakeasy frontend understands.

        Examples
        --------
        >>> MultimediaIndex.to_speakeasy_token("0000/abc123")
        'image:0000/abc123'
        """
        return f"image:{image_id}"

    def tokens_for_uri(self, uri: str, max_images: int = 1) -> List[str]:
        """
        Return up to `max_images` image tokens for the given entity URI.

        These tokens can be directly included in the text reply and will
        be rendered as images in the UI.
        """
        image_ids = self.get_image_ids_for_uri(uri)
        if not image_ids:
            return []
        return [self.to_speakeasy_token(img_id) for img_id in image_ids[:max_images]]

    def batch_tokens_for_uris(
        self,
        uris: Iterable[str],
        max_images_per_entity: int = 1,
    ) -> Dict[str, List[str]]:
        """
        For each URI in `uris`, return a list of image tokens.

        Result is a mapping uri -> [token, ...]. URIs without images are
        omitted from the mapping.
        """
        result: Dict[str, List[str]] = {}
        for uri in uris:
            tokens = self.tokens_for_uri(uri, max_images=max_images_per_entity)
            if tokens:
                result[uri] = tokens
        return result


# Module-level singleton, same pattern as for GraphExecutor
_GLOBAL_MM_INDEX: Optional[MultimediaIndex] = None


def get_global_multimedia_index() -> MultimediaIndex:
    """
    Return a process-wide singleton MultimediaIndex.
    """
    global _GLOBAL_MM_INDEX
    if _GLOBAL_MM_INDEX is None:
        _GLOBAL_MM_INDEX = MultimediaIndex()
    return _GLOBAL_MM_INDEX
