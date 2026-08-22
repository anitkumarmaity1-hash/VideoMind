"""
Visual embedding generation using OpenCLIP.

Supports embedding both images (frames) and text queries into the SAME
CLIP embedding space, which is what makes text->visual retrieval possible
("what objects appear in the video?" -> query CLIP text tower -> match
against CLIP image embeddings of sampled frames).
"""
from typing import List, Optional, Tuple, Any
from PIL import Image
from app.config import settings

_model: Optional[Any] = None
_preprocess: Optional[Any] = None
_tokenizer: Optional[Any] = None
_device: Optional[str] = None


def _load() -> Tuple[Any, Any, Any, str]:
    """Lazily loads and caches the CLIP model/preprocess/tokenizer/device.

    Return type is annotated as non-Optional (unlike the module-level
    globals, which start as None) because this function always populates
    them before returning — the `assert`s make that guarantee explicit
    for both Pylance and anyone reading this at 2am, rather than callers
    having to re-null-check something that's actually always set.
    """
    global _model, _preprocess, _tokenizer, _device
    if _model is None:
        import torch
        import open_clip

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            settings.visual_embedding_model,
            pretrained=settings.visual_embedding_pretrained,
        )
        _model.to(_device)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(settings.visual_embedding_model)

    assert _model is not None and _preprocess is not None and _tokenizer is not None and _device is not None
    return _model, _preprocess, _tokenizer, _device


def embed_images(image_paths: List[str], batch_size: int = 16) -> List[List[float]]:
    """Batches images through CLIP instead of one-by-one. On CPU especially,
    per-call Python/tensor overhead dominates for single-image forward
    passes; batching amortizes that across `batch_size` images per call."""
    import torch

    model, preprocess, _tokenizer, device = _load()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            ).to(device)
            feats = model.encode_image(batch_tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.extend(feats.cpu().numpy().tolist())
    return embeddings


def embed_text_for_visual_search(text: str) -> List[float]:
    """Embed a natural-language query into CLIP's joint embedding space,
    used to retrieve visually-relevant frames for a question."""
    import torch

    model, _preprocess, tokenizer, device = _load()
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy().tolist()
