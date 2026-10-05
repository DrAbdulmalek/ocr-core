"""Unified golden sample dataset — deterministic, self-owned pairs.

docs/09 §6 item 1: "عينة ذهبية موحدة" — the SAME sample (30+ Arabic/English
pairs) passed through every engine with the SAME normalization, producing
our first owned comparative CER table.

Design rules (binding for any future edit of this file):
  1. Determinism: every pair carries an explicit ``seed``; the renderer must
     be a pure function of (pair, font file). No global RNG state.
  2. Reproducibility: pairs are regenerable from this module alone — no
     binary images are committed; nothing can be "lost" again (the owner's
     documented complaint about Telegram-page work disappearing).
  3. Originality: every text below is original clinical-style prose written
     for this benchmark — no copyrighted excerpt is embedded.
  4. Coverage: both languages and all four degradation conditions
     (clean / shadow / low_contrast / noisy) appear; the shadow condition is
     the documented Otsu-collapse case from PR #14's proof-of-idea test.
  5. Minimum size: ≥ 30 pairs (docs/09 §6). Currently 32.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GoldenPair", "CONDITIONS", "MIN_PAIRS", "golden_pairs"]

#: degradation conditions exercised by the benchmark
CONDITIONS = ("clean", "shadow", "low_contrast", "noisy")

#: docs/09 §6 requires at least 30 pairs
MIN_PAIRS = 30


@dataclass(frozen=True)
class GoldenPair:
    """One golden sample: ground-truth text + rendering parameters."""

    id: str
    lang: str          # "ar" | "en"
    condition: str     # one of CONDITIONS
    seed: int
    font_px: int
    text: str


_AR_TEXTS: list[tuple[str, str, int]] = [
    # (condition, text, font_px) — condition is explicit per pair
    ("clean", "المريض يعاني من ارتفاع ضغط الدم المزمن منذ خمس سنوات.\nتقرر وصف أملوديبين بجرعة 5 ملغ يوميا مع مراقبة الضغط اسبوعيا.", 40),
    ("shadow", "اظهرت اشعة الصدر اعتلالا طفيفا في الفص السفلي الايمن.\nيوصى باعادة التصوير بعد اسبوعين لتقييم تطور الحالة.", 38),
    ("low_contrast", "كشف تحليل الدم عن فقر دم بعوز الحديد.\nالهيموغلوبين 9.8 غم لكل ديسيلتر يستدعي علاجا تعويضيا لمدة ثلاثة اشهر.", 40),
    ("noisy", "تاكد تشخيص داء السكري من النوع الثاني بمستوى سكر صائم 162 ملغ.\nيوصى بميتفورمين 850 ملغ مرتين يوميا مع حمية غذائية مناسبة.", 36),
    ("clean", "التصوير بالموجات فوق الصوتية اظهر حصاة قنوية بحجم 7 مم.\nلا يوجد توسع للقنوات الصفراوية ويكفي المتابعة المحافظة.", 42),
    ("shadow", "شكاوى المريض من سعال جاف مستمر منذ ثلاثة اسابيع وليلي.\nالفحص السريري سليم عما عدا حرارة خفيفة 37.8 درجة.", 38),
    ("low_contrast", "وظائف الغدة الدرقية ضمن الحدود الطبيعية.\nالهرمون المنبه للدرقية 2.1 وحدة دولية لكل لتر.", 40),
    ("noisy", "الرسم القلبي يظهر نظما جيبيا منتظما بمعدل 72 نبضة في الدقيقة.\nلا توجد علامات نقص تروية حديثة.", 36),
    ("clean", "امراض المريض السابقة تشمل الربو الشعبي وارتفاع الشحوم.\nليست لديه حساسية دوائية معروفة حتى تاريخه.", 40),
    ("shadow", "التصوير الطبقي المحوري للبطن يظهر كبد سليم الحجم.\nالطرق الصفراوية غير متوسعة والطحال بمقاسات طبيعية.", 38),
    ("low_contrast", "استئصال المرارة بالمنظار تم بنجاح دون مضاعفات.\nخروج المريض من المستشفى متوقع خلال يومين.", 40),
    ("noisy", "المزرعة البولية اظهرت نمو اشريكية قولونية حساسة للسيفترياكسون.\nيوصى بجرعة 1 غم وريديا كل اثنتي عشرة ساعة.", 36),
    ("clean", "ضغط الدم عند القياس الاخير 135 على 85 ملم زئبق.\nمؤشر كتلة الجسم 27.4 يصنف زيادة وزن درجة اولى.", 42),
    ("shadow", "الفحص بالمنظار العلوي اظهر التهابا في المريء السفلي.\nيوصى باوميبرازول 20 ملغ صباحا لمدة اربعة اسابيع.", 38),
    ("low_contrast", "مستوى البوتاسيوم في الدم 3.2 مليمول لكل لتر.\nيستلزم الامر تعويضا وريديا مع مراقبة تخطيط القلب.", 40),
    ("noisy", "عظم الفخذ الايمن سليم في الصورة الشعاعية.\nالمفصل الحقاني محفوظ ولا يوجد كسر واضح.", 36),
    ("clean", "خطة العلاج الطبيعي ثلاث جلسات اسبوعيا لمدة شهر.\nالهدف تحسين مدى الحركة في الركبة اليسرى.", 40),
    ("shadow", "تاريخ العائلة يذكر داء سكري نمط ثان عند الام.\nالجد توفي بسبب قلبي في سن الخمسين.", 38),
    ("low_contrast", "وظائف الكبد ضمن الطبيعي عما عدا ارتفاع طفيف بمقدار 46 وحدة.\nيوصى باعادة التحليل بعد ايقاف الادوية الحالية.", 40),
    ("noisy", "تم تركيب جهاز تنظيم ضربات القلب بنجاح.\nالايقاع المفروض VVI بمعدل اساسي 60 نبضة.", 36),
]

_EN_TEXTS: list[tuple[str, str, int]] = [
    ("clean", "The patient presents with chronic hypertension for five years.\nAmlodipine 5 mg once daily was prescribed with weekly monitoring.", 36),
    ("shadow", "Chest radiograph shows mild infiltration in the right lower lobe.\nFollow-up imaging is recommended within two weeks.", 34),
    ("low_contrast", "Blood analysis reveals iron deficiency anemia.\nHemoglobin level is 9.8 g/dL and requires supplementation.", 36),
    ("noisy", "Type 2 diabetes confirmed with fasting glucose of 162 mg/dL.\nMetformin 850 mg twice daily with dietary counseling.", 32),
    ("clean", "Abdominal ultrasound demonstrates a 7 mm common bile duct stone.\nNo biliary dilation is present and conservative follow-up suffices.", 36),
    ("shadow", "Upper endoscopy reveals inflammation of the lower esophagus.\nOmeprazole 20 mg daily for four weeks is advised.", 34),
    ("low_contrast", "Thyroid function tests are within normal limits.\nTSH level measured at 2.1 IU/L.", 36),
    ("noisy", "Electrocardiogram shows regular sinus rhythm at 72 beats per minute.\nNo evidence of acute ischemic changes.", 32),
    ("clean", "Blood culture grew Escherichia coli sensitive to ceftriaxone.\nRecommended dose is 1 g intravenously every twelve hours.", 36),
    ("shadow", "Serum potassium is 3.2 mmol/L.\nIntravenous replacement with cardiac monitoring is required.", 34),
    ("low_contrast", "Laparoscopic cholecystectomy was completed without complications.\nDischarge is expected within two days.", 36),
    ("noisy", "The patient was discharged home in stable condition.\nFollow-up appointment scheduled in the cardiology clinic after ten days.", 32),
]


def golden_pairs() -> list[GoldenPair]:
    """Return the frozen golden sample (deterministic order: ar then en)."""
    pairs: list[GoldenPair] = []
    for i, (condition, text, font_px) in enumerate(_AR_TEXTS, start=1):
        pairs.append(
            GoldenPair(
                id=f"ar-{i:03d}",
                lang="ar",
                condition=condition,
                seed=1000 + i,
                font_px=font_px,
                text=text,
            )
        )
    for i, (condition, text, font_px) in enumerate(_EN_TEXTS, start=1):
        pairs.append(
            GoldenPair(
                id=f"en-{i:03d}",
                lang="en",
                condition=condition,
                seed=2000 + i,
                font_px=font_px,
                text=text,
            )
        )
    return pairs
