| field | value |
|---|---|
| engines | tesseract |
| variants | raw, otsu, sauvola, auto |
| pairs | 32 |
| normalization | arabic_strong_normalize (both sides) |



### Overall — mean CER/WER per engine × variant

| engine | variant | n | mean CER | mean WER | errors |
|---|---|---|---|---|---|
| tesseract | raw | 32 | 0.307054 | 1.809067 | 0 |
| tesseract | otsu | 32 | 0.310848 | 1.825201 | 0 |
| tesseract | sauvola | 32 | 0.198003 | 1.130102 | 0 |
| tesseract | auto | 32 | 0.233981 | 1.336702 | 0 |



### By language

| engine | lang | variant | n | mean CER | mean WER |
|---|---|---|---|---|---|
| tesseract | ar | raw | 20 | 0.466602 | 2.711851 |
| tesseract | ar | otsu | 20 | 0.475247 | 2.758497 |
| tesseract | ar | sauvola | 20 | 0.307054 | 1.739741 |
| tesseract | ar | auto | 20 | 0.365038 | 2.072732 |
| tesseract | en | raw | 12 | 0.04114 | 0.304428 |
| tesseract | en | otsu | 12 | 0.03685 | 0.269706 |
| tesseract | en | sauvola | 12 | 0.01625 | 0.114035 |
| tesseract | en | auto | 12 | 0.015554 | 0.109984 |



### By page condition

| engine | condition | variant | n | mean CER | mean WER |
|---|---|---|---|---|---|
| tesseract | clean | raw | 8 | 0.202954 | 1.078536 |
| tesseract | clean | otsu | 8 | 0.202954 | 1.078536 |
| tesseract | clean | sauvola | 8 | 0.20048 | 1.097333 |
| tesseract | clean | auto | 8 | 0.202954 | 1.078536 |
| tesseract | low_contrast | raw | 8 | 0.086817 | 0.528156 |
| tesseract | low_contrast | otsu | 8 | 0.073886 | 0.447799 |
| tesseract | low_contrast | sauvola | 8 | 0.012069 | 0.08349 |
| tesseract | low_contrast | auto | 8 | 0.073886 | 0.447799 |
| tesseract | noisy | raw | 8 | 0.403548 | 2.332509 |
| tesseract | noisy | otsu | 8 | 0.413803 | 2.380924 |
| tesseract | noisy | sauvola | 8 | 0.33418 | 1.900035 |
| tesseract | noisy | auto | 8 | 0.413803 | 2.380924 |
| tesseract | shadow | raw | 8 | 0.534896 | 3.297068 |
| tesseract | shadow | otsu | 8 | 0.552749 | 3.393544 |
| tesseract | shadow | sauvola | 8 | 0.245282 | 1.439548 |
| tesseract | shadow | auto | 8 | 0.245282 | 1.439548 |



### Best variant per condition (min mean CER)

| engine | condition | best variant | mean CER |
|---|---|---|---|
| tesseract | clean | sauvola | 0.20048 |
| tesseract | low_contrast | sauvola | 0.012069 |
| tesseract | noisy | sauvola | 0.33418 |
| tesseract | shadow | sauvola | 0.245282 |
