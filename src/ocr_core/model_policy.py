"""سياسة الوصول للنماذج — محلي-فقط افتراضيًا (OCR-05/#79، تتبّع #77 و IFM#89).

مفردات صغيرة بلا أي تبعيات تحكم من أين تُحلَّل أوزان النماذج:

- ``LOCAL_ONLY`` (الافتراضي): الحل من الخزنة المحلية فقط — أي نقص أثر
  يفشل مغلقًا (``ModelArtifactMissingError``) **دون أي محاولة شبكة**.
- ``EXPLICIT_REMOTE``: الشبكة مسموحة فقط عند ``remote_authorized=True``
  (صريح من المالك) — محاولة واحدة، لا إعادة محاولة على فشل الخزنة.
- ``REMOTE_DISABLED``: أبعد من ذلك — لا شبكة حتى مع التصريح.

المحاكم: بارامترات المُنشئ > متغيرات البيئة ``OCR_CORE_MODEL_POLICY``
و``OCR_CORE_REMOTE_AUTHORIZED`` > الافتراضي ``LOCAL_ONLY``.

فلسفة متسقة مع قاعدة الخصوصية (ocr-core#76 / خطة الدمج §8.3): لا شبكة،
لا تنزيلات صامتة، فشل مغلق مع رسالة قابلة للتنفيذ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# ─── المفردات المشتركة (تُستشهد بها القضايا #77/#79 و IFM#89) ────────────────
LOCAL_ONLY = "LOCAL_ONLY"
EXPLICIT_REMOTE = "EXPLICIT_REMOTE"
REMOTE_DISABLED = "REMOTE_DISABLED"
POLICY_MODES: Tuple[str, ...] = (LOCAL_ONLY, EXPLICIT_REMOTE, REMOTE_DISABLED)

ENV_POLICY = "OCR_CORE_MODEL_POLICY"
ENV_AUTHORIZED = "OCR_CORE_REMOTE_AUTHORIZED"


class ModelArtifactMissingError(RuntimeError):
    """أثر النموذج غير موجود في الخزنة المحلية والوصول البعيد غير مصرّح.

    الرسالة قابلة للتنفيذ دائمًا: كيف تجعل الأثر متاحًا محليًا أو كيف
    تُصرّح بالوصول البعيد صراحةً إن كنت المالك وتريد ذلك.
    """


def _parse_bool_env(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ModelAccessPolicy:
    """قرار الوصول: أي وضع، وهل التصريح بالبعيد قائم.

    ``resolve()`` يطبّق التسلسل: بارامتر صريح > متغير بيئة > الافتراضي.
    وضع غير معروف = ``ValueError`` (فشل مغلق — لا تخمين صامت).
    """

    mode: str = LOCAL_ONLY
    remote_authorized: bool = False

    def __post_init__(self):
        if self.mode not in POLICY_MODES:
            raise ValueError(
                f"وضع سياسة غير معروف: {self.mode!r} — المسموح: {POLICY_MODES}"
            )

    @classmethod
    def resolve(
        cls,
        mode: Optional[str] = None,
        remote_authorized: Optional[bool] = None,
    ) -> "ModelAccessPolicy":
        """يحسم الوضع النهائي: بارامتر > بيئة > ``LOCAL_ONLY``."""
        resolved_mode = mode if mode is not None else os.environ.get(ENV_POLICY, LOCAL_ONLY).strip()
        resolved_auth = (
            remote_authorized
            if remote_authorized is not None
            else _parse_bool_env(ENV_AUTHORIZED)
        )
        return cls(mode=resolved_mode, remote_authorized=bool(resolved_auth))

    def allows_remote(self) -> bool:
        """الشبكة مسموحة فقط في ``EXPLICIT_REMOTE`` مع تصريح صريح."""
        return self.mode == EXPLICIT_REMOTE and self.remote_authorized

    def describe(self) -> dict:
        """وصف عدّادات لا محتوى — للتقارير والتشخيص."""
        return {
            "mode": self.mode,
            "remote_authorized": self.remote_authorized,
            "allows_remote": self.allows_remote(),
            "env": {ENV_POLICY: os.environ.get(ENV_POLICY, ""),
                    ENV_AUTHORIZED: os.environ.get(ENV_AUTHORIZED, "")},
        }


def artifact_missing_message(model_id: str, cache_dir) -> str:
    """رسالة موحّدة قابلة للتنفيذ لفشل الأثر المحلي."""
    return (
        f"أثر النموذج {model_id} غير مخزّن محليًا (الخزنة: {cache_dir}) "
        f"والسياسة الافتراضية {LOCAL_ONLY} تمنع أي تنزيل شبكي. الخيارات:\n"
        f"  (أ) اجعل الأثر متاحًا في الخزنة محليًا (نسخة يدوية/داخلية).\n"
        f"  (ب) صرّح بالوصول البعيد صراحةً: {ENV_POLICY}={EXPLICIT_REMOTE} "
        f"و{ENV_AUTHORIZED}=1 — أو بارامتر policy/remote_authorized في المُنشئ."
    )
