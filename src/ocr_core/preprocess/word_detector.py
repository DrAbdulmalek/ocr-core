"""كشف الكلمات في الصور المكتوبة بخط اليد.

يوفر 3 كواشف قابلة للتبديل:
  1. OnnxWordDetector     — نموذج xournalpp-htr (دقيق، يحتاج onnxruntime)
  2. ProjectionDetector   — إسقاط عمودي (سريع، جيد للخط المتصل)
  3. HeuristicDetector    — كشف بالكثافة (بدون نموذج، للطوارئ)

الواجهة الموحّدة: `detect(image) -> list[BBox]`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


# ---------- نموذج المربع ----------
@dataclass
class BBox:
    """مربع محيط بكلمة — في إحداثيات الصورة الأصلية."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0
    source: str = "unknown"        # أي كاشف أنتجه
    line_index: Optional[int] = None
    word_index: Optional[int] = None

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_dict(self) -> dict:
        return {
            "bbox": [self.x_min, self.y_min, self.x_max, self.y_max],
            "confidence": self.confidence,
            "source": self.source,
        }


# ---------- الواجهة الموحّدة ----------
class WordDetector(Protocol):
    """عقد كل كاشف كلمات."""
    name: str

    def detect(self, image) -> list[BBox]:
        """يستقبل PIL.Image ويرجع قائمة BBox."""
        ...


# ==============================================================
# 1. كاشف ONNX — xournalpp-htr-word-detector
# ==============================================================
class OnnxWordDetector:
    """كاشف يعتمد على نموذج xournalpp-htr-word-detector (MIT).

    النموذج: ResNet-18 encoder + U-Net decoder
    الإدخال: صورة رمادية 448×448
    الإخراج: 7 خرائط ميزات → مربعات كلمات
    الترخيص: MIT
    """

    name = "onnx"

    def __init__(
        self,
        model_id: str = "PellelNitram/xournalpp-htr-word-detector",
        min_confidence: float = 0.3,
        cache_dir: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.min_confidence = min_confidence
        self.cache_dir = cache_dir

        self._session = None
        self._loaded = False

    def _ensure_loaded(self):
        """تحميل النموذج كسولاً — لا يحمّل شيئًا حتى أول استدعاء."""
        if self._loaded:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "pip install onnxruntime huggingface_hub\n"
                "أو: pip install xournalpp-htr"
            )

        # حمّل النموذج من HuggingFace (مع تخزين مؤقت)
        model_path = self._download_model()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._loaded = True
        logger.info("تم تحميل WordDetector من %s", self.model_id)

    def _download_model(self) -> Path:
        """ينزّل النموذج إلى cache إن لم يكن موجودًا."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        cache = self.cache_dir or Path.home() / ".cache" / "ocr_core" / "models"
        cache.mkdir(parents=True, exist_ok=True)

        path = hf_hub_download(
            repo_id=self.model_id,
            filename="model.onnx",
            cache_dir=str(cache),
        )
        return Path(path)

    def detect(self, image) -> list[BBox]:
        """يكشف الكلمات ويعيد BBox بإحداثيات الصورة الأصلية."""
        self._ensure_loaded()

        import numpy as np
        from PIL import Image

        # 1. حوّل إلى رمادي
        if image.mode != "L":
            image = image.convert("L")

        original_size = image.size
        orig_w, orig_h = original_size

        # 2. غيّر الحجم إلى 448×448
        resized = image.resize((448, 448), Image.LANCZOS)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=(0, 1))  # (1, 1, 448, 448)

        # 3. شغّل النموذج
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: arr})

        # 4. استخرج المربعات من الخرائط
        boxes_448 = self._decode_outputs(outputs)

        # 5. أعد الحجم إلى الأصل
        scale_x = orig_w / 448.0
        scale_y = orig_h / 448.0

        boxes = []
        for b in boxes_448:
            x1 = max(0, int(b.x_min * scale_x))
            y1 = max(0, int(b.y_min * scale_y))
            x2 = min(orig_w, int(b.x_max * scale_x))
            y2 = min(orig_h, int(b.y_max * scale_y))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            boxes.append(BBox(
                x_min=x1, y_min=y1, x_max=x2, y_max=y2,
                confidence=b.confidence,
                source=self.name,
            ))

        # فلترة بالثقة
        boxes = [b for b in boxes if b.confidence >= self.min_confidence]
        return boxes

    @staticmethod
    def _decode_outputs(outputs: list) -> list[BBox]:
        """فكّ 7 خرائط ميزات إلى قائمة مربعات.

        البنية الفعلية للنموذج قد تختلف قليلاً. هذه دالة أولية
        تستخدم خريطة "segmentation" لاستخراج المناطق الملوّنة
        ثم تحسب المربعات المحيطة بها.
        """
        import numpy as np

        # outputs[0] عادة: (1, 7, 224, 224) أو مشابه
        out = outputs[0]
        if out.ndim == 4:
            out = out[0]

        # أول 3 قنوات = segmentation (word / margin / background)
        # softmax مُدمج مسبقًا
        if out.shape[0] >= 3:
            word_map = out[0]
        else:
            word_map = out.squeeze()

        # عتبة للكشف
        mask = (word_map > 0.5).astype(np.uint8) * 255

        # ابحث عن المكونات المتصلة
        try:
            import cv2
            num_labels, labels = cv2.connectedComponents(mask)
            boxes = []
            for label_id in range(1, num_labels):
                ys, xs = np.where(labels == label_id)
                if len(xs) < 20:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                # حوّل من 224 إلى 448 (المخرجات بنصف دقة المدخل)
                boxes.append(BBox(
                    x_min=x1 * 2, y_min=y1 * 2,
                    x_max=x2 * 2, y_max=y2 * 2,
                    confidence=float(word_map[ys, xs].mean()),
                    source="onnx",
                ))
            return boxes
        except ImportError:
            # fallback: استخدام bounding box كامل
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                return []
            return [BBox(
                x_min=int(xs.min()), y_min=int(ys.min()),
                x_max=int(xs.max()), y_max=int(ys.max()),
                confidence=0.5, source="onnx",
            )]


# ==============================================================
# 2. كاشف الإسقاط — سريع، جيد للعربية المتصلة
# ==============================================================
class ProjectionDetector:
    """كشف الكلمات بالإسقاط العمودي — مناسب خاصة للخط المتصل.

    الفكرة: النص العربي متصل داخل الكلمة، لكن هناك فجوات بيضاء
    بين الكلمات. الإسقاط العمودي (عدد البكسلات السوداء في كل عمود)
    يُظهر قممًا داخل الكلمات وفجوات بينها.
    """

    name = "projection"

    def __init__(
        self,
        darkness_threshold: int = 128,
        min_gap_width: int = 8,
        min_word_width: int = 15,
        min_word_height: int = 10,
        vertical_fill_ratio: float = 0.02,
    ):
        self.darkness_threshold = darkness_threshold
        self.min_gap_width = min_gap_width
        self.min_word_width = min_word_width
        self.min_word_height = min_word_height
        self.vertical_fill_ratio = vertical_fill_ratio

    def detect(self, image) -> list[BBox]:
        import numpy as np

        if image.mode != "L":
            image = image.convert("L")

        arr = np.array(image)
        h, w = arr.shape
        if h == 0 or w == 0:
            return []

        # 1. عتبة: كم بكسل داكن في كل عمود؟
        dark = arr < self.darkness_threshold
        col_counts = dark.sum(axis=0)
        threshold = max(1, int(h * self.vertical_fill_ratio))

        # 2. ابحث عن المجموعات (كلمات)
        boxes = []
        in_word = False
        start = 0
        gap_count = 0

        for i, count in enumerate(col_counts):
            if count > threshold:
                if not in_word:
                    in_word = True
                    start = i
                gap_count = 0
            else:
                if in_word:
                    gap_count += 1
                    if gap_count >= self.min_gap_width:
                        # انتهت الكلمة
                        if i - gap_count - start >= self.min_word_width:
                            boxes.append(self._make_bbox(
                                dark, start, i - gap_count, h, w,
                            ))
                        in_word = False
                        gap_count = 0

        # كلمة أخيرة
        if in_word and (w - start) >= self.min_word_width:
            boxes.append(self._make_bbox(dark, start, w, h, w))

        return [b for b in boxes if b is not None]

    def _make_bbox(self, dark_mask, x1: int, x2: int,
                    h: int, w: int) -> Optional[BBox]:
        """يبني BBox لكلمة بين أعمدة [x1, x2)."""
        import numpy as np  # إصلاح تدقيق: كان np مستخدمًا بلا استيراد في هذا النطاق

        if x2 <= x1:
            return None
        column_slice = dark_mask[:, x1:x2]
        rows_with_content = np.where(column_slice.any(axis=1))[0]
        if len(rows_with_content) == 0:
            return None
        y1 = int(rows_with_content.min())
        y2 = int(rows_with_content.max()) + 1
        if y2 - y1 < self.min_word_height:
            return None
        return BBox(
            x_min=x1, y_min=y1, x_max=x2, y_max=y2,
            confidence=0.7, source=self.name,
        )


# ==============================================================
# 3. كاشف Heuristic — للطوارئ فقط
# ==============================================================
class HeuristicDetector:
    """كاشف بسيط جدًا — يقسّم الصورة إلى أشرطة أفقية.

    لا يُستخدم للإنتاج. موجود فقط كخطة بديلة عندما تكون
    كل النماذج مفقودة.
    """

    name = "heuristic"

    def __init__(self, strip_height: int = 60, strip_gap: int = 10):
        self.strip_height = strip_height
        self.strip_gap = strip_gap

    def detect(self, image) -> list[BBox]:
        w, h = image.size
        boxes = []
        y = 0
        while y < h:
            y2 = min(y + self.strip_height, h)
            boxes.append(BBox(
                x_min=0, y_min=y, x_max=w, y_max=y2,
                confidence=0.3, source=self.name,
            ))
            y = y2 + self.strip_gap
        return boxes


# ==============================================================
# المصنع — يختار الكاشف المناسب
# ==============================================================
def create_detector(
    kind: str = "auto",
    **kwargs,
) -> WordDetector:
    """ينشئ كاشفًا حسب النوع.

    kinds:
      "onnx"       — الأفضل جودة، يحتاج onnxruntime
      "projection" — الأسرع، بلا تبعيات ثقيلة
      "heuristic"  — للطوارئ فقط
      "auto"       — يحاول onnx ثم projection ثم heuristic
    """
    if kind == "auto":
        try:
            import onnxruntime  # noqa
            return OnnxWordDetector(**kwargs)
        except ImportError:
            logger.info("onnxruntime غير مثبت — استخدام projection")
            return ProjectionDetector(**kwargs)

    if kind == "onnx":
        return OnnxWordDetector(**kwargs)
    if kind == "projection":
        return ProjectionDetector(**kwargs)
    if kind == "heuristic":
        return HeuristicDetector(**kwargs)

    raise ValueError(f"نوع كاشف غير معروف: {kind}")


def detect_words(image, kind: str = "auto", **kwargs) -> list[BBox]:
    """اختصار: كشف الكلمات في صورة."""
    detector = create_detector(kind, **kwargs)
    return detector.detect(image)
