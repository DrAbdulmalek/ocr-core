"""أوامر المعالجة المسبقة — بعضها مستخلص من photocraft."""
from ..core import command, ExecutionContext


@command(
    id="preprocess.deskew", name="تصحيح الميل",
    description="يصحح ميل الصورة تلقائياً.",
    category="preprocess",
    params_schema={"type": "object", "properties": {"input_key": {"type": "string"}}, "required": []},
)
def deskew(ctx: ExecutionContext, input_key: str = "image") -> dict:
    import cv2, numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    if img is None:
        raise ValueError(f"لا صورة بمفتاح '{input_key}'")
    arr = np.array(img.convert("L"))
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return {"angle": 0, "skipped": True}
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    ctx.set(input_key, Image.fromarray(rotated))
    return {"angle": round(angle, 2)}


@command(
    id="preprocess.denoise", name="إزالة الضوضاء",
    description="ينظّف الصورة من الضجيج الرقمي.",
    category="preprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "strength": {"type": "integer"}}, "required": []},
)
def denoise(ctx: ExecutionContext, input_key: str = "image", strength: int = 10) -> dict:
    import cv2, numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    arr = np.array(img.convert("L"))
    denoised = cv2.fastNlMeansDenoising(arr, None, strength, 7, 21)
    ctx.set(input_key, Image.fromarray(denoised))
    return {"strength": strength}


@command(
    id="preprocess.levels", name="ضبط المستويات",
    description="Levels: black/white/gamma — مستخلص من photocraft.",
    category="preprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "black": {"type": "integer"},
        "white": {"type": "integer"}, "gamma": {"type": "number"}}, "required": []},
)
def levels(ctx: ExecutionContext, input_key: str = "image",
            black: int = 0, white: int = 255, gamma: float = 1.0) -> dict:
    import numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    arr = np.array(img.convert("L")).astype(np.float32)
    arr = (arr - black) / max(white - black, 1) * 255.0
    arr = np.clip(arr, 0, 255)
    if gamma != 1.0:
        arr = np.power(arr / 255.0, 1.0 / gamma) * 255.0
    ctx.set(input_key, Image.fromarray(arr.astype(np.uint8)))
    return {"black": black, "white": white, "gamma": gamma}


@command(
    id="preprocess.curves", name="ضبط المنحنيات",
    description="Curves مع نقاط تحكم — مستخلص من photocraft.",
    category="preprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "control_points": {"type": "array"}},
        "required": ["control_points"]},
)
def curves(ctx: ExecutionContext, input_key: str = "image", control_points: list = None) -> dict:
    import numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    arr = np.array(img.convert("L"))
    pts = control_points or [(0, 0), (255, 255)]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    ctx.set(input_key, Image.fromarray(lut[arr]))
    return {"points": len(pts)}


@command(
    id="preprocess.histogram_equalization", name="معادلة الهيستوغرام",
    description="CLAHE — تحسين تباين متقدم.",
    category="preprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "clip_limit": {"type": "number"}}, "required": []},
)
def histogram_equalization(ctx: ExecutionContext, input_key: str = "image",
                            clip_limit: float = 2.0) -> dict:
    import cv2, numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    arr = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    ctx.set(input_key, Image.fromarray(clahe.apply(arr)))
    return {"clip_limit": clip_limit}


@command(
    id="preprocess.binarize", name="تحويل ثنائي",
    description="يحوّل الصورة إلى أبيض/أسود (Otsu أو Adaptive).",
    category="preprocess",
    params_schema={"type": "object", "properties": {
        "input_key": {"type": "string"}, "method": {"type": "string"}}, "required": []},
)
def binarize(ctx: ExecutionContext, input_key: str = "image", method: str = "otsu") -> dict:
    import cv2, numpy as np
    from PIL import Image
    img = ctx.get(input_key)
    arr = np.array(img.convert("L"))
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ctx.set(input_key, Image.fromarray(binary))
    return {"method": method}
