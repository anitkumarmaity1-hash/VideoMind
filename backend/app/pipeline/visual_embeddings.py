"""
Visual embedding generation using OpenCLIP.

Supports embedding both images (frames) and text queries into the SAME
CLIP embedding space, which is what makes text->visual retrieval possible
("what objects appear in the video?" -> query CLIP text tower -> match
against CLIP image embeddings of sampled frames).
"""
from typing import List
from PIL import Image
from app.config import settings

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load():
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
    return _model, _preprocess, _tokenizer, _device


def embed_images(image_paths: List[str]) -> List[List[float]]:
    import torch

    model, preprocess, _tokenizer, device = _load()
    embeddings = []
    with torch.no_grad():
        for path in image_paths:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat.squeeze(0).cpu().numpy().tolist())
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
