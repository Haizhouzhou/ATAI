import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class MultimediaIndex:
    """
    Handles looking up images from the images.json file.
    Fix: STRICTLY selects only .jpg/.jpeg images to avoid broken PNGs in frontend.
    """
    def __init__(self, images_json_path: Path, metadata_dir: Path = None):
        self.images_path = images_json_path
        self.image_map: Dict[str, str] = {} 
        self.imdb_map: Dict[str, str] = {}
        
        # Load IMDb map for bridging
        if metadata_dir:
            imdb_path = metadata_dir / "imdb_map.json"
            if imdb_path.exists():
                try:
                    with open(imdb_path, "r") as f: 
                        self.imdb_map = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load imdb_map: {e}")
        
        self._load_index()

    def _load_index(self):
        if not self.images_path.exists():
            return

        try:
            with open(self.images_path, "r", encoding="utf-8") as f: 
                data = json.load(f)
            
            count = 0
            for item in data:
                img_path = item.get("img")
                if not img_path: continue
                
                # --- CRITICAL FIX: Only allow JPGs ---
                # The frontend fails with PNGs even if ID is stripped.
                # We prioritize JPGs which align with the assignment examples.
                if not img_path.lower().endswith((".jpg", ".jpeg")):
                    continue
                # -------------------------------------
                
                # Remove extension for ID
                clean_id, _ = os.path.splitext(img_path)
                
                # Map URIs
                uris = []
                for key in ["movie", "cast", "id"]:
                    val = item.get(key)
                    if isinstance(val, list): 
                        uris.extend(val)
                    elif isinstance(val, str): 
                        uris.append(val)
                
                for uri in uris:
                    clean_uri = uri.strip("<>")
                    self.image_map[clean_uri] = clean_id
                    
                    if clean_uri.startswith("tt") or clean_uri.startswith("nm"):
                        self.image_map[clean_uri] = clean_id
                    
                    count += 1
            
            logger.info(f"Loaded {count} JPG image mappings.")
            
        except Exception as e:
            logger.error(f"Failed to load images.json: {e}", exc_info=True)

    def get_image(self, uri: str) -> Optional[str]:
        clean = uri.strip("<>")
        
        # 1. Direct Match
        if clean in self.image_map: 
            return f"image:{self.image_map[clean]}"
        
        # 2. IMDb Bridge Match
        if clean in self.imdb_map:
            imdb_id = self.imdb_map[clean]
            if imdb_id in self.image_map: 
                return f"image:{self.image_map[imdb_id]}"
                
        return None