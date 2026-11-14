# Duplicate Groups Representative Selection Report

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
**Dataset:** Primary Duplicate-Collapsed Benchmark (703 Images)

## 1. Selection Methodology
For each of the 35 same-class multi-image duplicate groups identified by SHA-256 and perceptual hash clustering (pHash Hamming distance $\le 6$), exactly one representative image was retained in the primary experimental dataset using an objective, deterministic rule:
1. **Primary Criterion:** Highest image resolution (total pixels: $\text{width} \times \text{height}$).
2. **Secondary Tie-Breaker:** Largest file size in bytes (indicative of lower compression artifacting).
3. **Tertiary Tie-Breaker:** Lexicographically smallest filename (e.g. `019.jpeg` before `449.jpeg`).

This deterministic selection is completely independent of model performance and was conducted without touching validation or test evaluations.

## 2. Summary of 35 Multi-Image Duplicate Groups and Representatives

| Group ID | Class | Split | Total Copies | Retained Representative | Resolution | File Size | Excluded Redundant Copies | Selection Reason |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- | :--- |
| **2** | CANCER | `train` | 2 | **`003.jpeg`** | 280x205 | 41,912 B | `486.jpeg` (280x205) | Equal resolution & size; alphabetical filename tie-breaker |
| **10** | CANCER | `train` | 2 | **`011.jpeg`** | 280x205 | 47,158 B | `432.jpeg` (280x205) | Equal resolution & size; alphabetical filename tie-breaker |
| **18** | CANCER | `train` | 2 | **`019.jpeg`** | 280x205 | 49,582 B | `449.jpeg` (280x205) | Equal resolution & size; alphabetical filename tie-breaker |
| **19** | CANCER | `train` | 2 | **`020.jpeg`** | 280x205 | 45,550 B | `440.jpeg` (280x205) | Equal resolution & size; alphabetical filename tie-breaker |
| **24** | CANCER | `val` | 2 | **`025.jpeg`** | 339x301 | 49,714 B | `054.jpeg` (339x301) | Equal resolution & size; alphabetical filename tie-breaker |
| **28** | CANCER | `train` | 2 | **`029.jpeg`** | 354x237 | 15,920 B | `055.jpeg` (354x237) | Equal resolution & size; alphabetical filename tie-breaker |
| **29** | CANCER | `train` | 2 | **`030.jpeg`** | 380x312 | 14,340 B | `056.jpeg` (380x312) | Equal resolution & size; alphabetical filename tie-breaker |
| **38** | CANCER | `train` | 3 | **`039.jpeg`** | 620x415 | 176,338 B | `057.jpeg` (620x415), `099.jpeg` (620x415) | Equal resolution & size; alphabetical filename tie-breaker |
| **40** | CANCER | `train` | 2 | **`041.jpeg`** | 3198x2166 | 4,071,097 B | `073.jpeg` (273x185) | Higher resolution (3198x2166 vs max 273x185) |
| **42** | CANCER | `train` | 2 | **`043.jpeg`** | 850x726 | 76,489 B | `058.jpeg` (850x726) | Equal resolution & size; alphabetical filename tie-breaker |
| **49** | CANCER | `train` | 3 | **`050.jpeg`** | 1105x950 | 597,187 B | `060.jpeg` (1105x950), `482.jpeg` (1105x950) | Equal resolution & size; alphabetical filename tie-breaker |
| **54** | CANCER | `train` | 2 | **`061.jpeg`** | 1936x1296 | 1,206,396 B | `079.jpeg` (1936x1296) | Equal resolution & size; alphabetical filename tie-breaker |
| **56** | CANCER | `val` | 2 | **`063.jpeg`** | 635x406 | 51,596 B | `092.jpeg` (635x388) | Higher resolution (635x406 vs max 635x388) |
| **57** | CANCER | `train` | 2 | **`064.jpeg`** | 375x279 | 185,841 B | `094.jpeg` (375x279) | Equal resolution & size; alphabetical filename tie-breaker |
| **58** | CANCER | `train` | 2 | **`065.jpeg`** | 375x287 | 194,004 B | `095.jpeg` (375x287) | Equal resolution & size; alphabetical filename tie-breaker |
| **60** | CANCER | `train` | 3 | **`067.jpeg`** | 698x467 | 28,245 B | `097.jpeg` (698x467), `434.jpeg` (250x168) | Equal resolution & size; alphabetical filename tie-breaker |
| **62** | CANCER | `train` | 2 | **`069.jpeg`** | 294x222 | 19,321 B | `101.jpeg` (294x222) | Equal resolution & size; alphabetical filename tie-breaker |
| **63** | CANCER | `train` | 3 | **`070.jpeg`** | 1100x618 | 64,155 B | `103.jpeg` (1100x618), `400.jpeg` (1100x618) | Equal resolution & size; alphabetical filename tie-breaker |
| **64** | CANCER | `train` | 2 | **`071.jpeg`** | 540x350 | 42,369 B | `107.jpeg` (540x350) | Equal resolution & size; alphabetical filename tie-breaker |
| **68** | CANCER | `train` | 2 | **`076.jpeg`** | 632x432 | 91,678 B | `199.jpeg` (632x432) | Equal resolution; larger file size (91,678 B vs 88,630 B) |
| **69** | CANCER | `train` | 2 | **`077.jpeg`** | 339x299 | 33,019 B | `083.jpeg` (239x211) | Higher resolution (339x299 vs max 239x211) |
| **75** | CANCER | `train` | 2 | **`093.jpeg`** | 1100x618 | 87,860 B | `085.jpeg` (300x168) | Higher resolution (1100x618 vs max 300x168) |
| **117** | CANCER | `train` | 2 | **`136.jpeg`** | 912x1056 | 103,550 B | `480.jpeg` (912x1056) | Equal resolution & size; alphabetical filename tie-breaker |
| **418** | CANCER | `train` | 2 | **`442.jpeg`** | 1200x900 | 279,086 B | `478.jpeg` (982x594) | Higher resolution (1200x900 vs max 982x594) |
| **421** | CANCER | `train` | 2 | **`494.jpeg`** | 550x371 | 31,950 B | `445.jpeg` (273x184) | Higher resolution (550x371 vs max 273x184) |
| **557** | NON CANCER | `train` | 2 | **`088.jpeg`** | 800x645 | 15,594 B | `165.jpeg` (250x201) | Higher resolution (800x645 vs max 250x201) |
| **602** | NON CANCER | `train` | 2 | **`133.jpeg`** | 800x769 | 41,802 B | `243.jpeg` (800x769) | Equal resolution; larger file size (41,802 B vs 40,939 B) |
| **618** | NON CANCER | `train` | 3 | **`149.jpeg`** | 612x402 | 28,756 B | `199.jpeg` (612x402), `211.jpeg` (277x182) | Equal resolution & size; alphabetical filename tie-breaker |
| **634** | NON CANCER | `train` | 2 | **`166.jpeg`** | 800x495 | 51,397 B | `196.jpeg` (680x422) | Higher resolution (800x495 vs max 680x422) |
| **635** | NON CANCER | `train` | 4 | **`175.jpeg`** | 781x1023 | 81,701 B | `187.jpeg` (700x917), `208.jpeg` (458x600), `167.jpeg` (344x450) | Higher resolution (781x1023 vs max 700x917) |
| **638** | NON CANCER | `test` | 4 | **`195.jpeg`** | 1500x1000 | 123,162 B | `179.jpeg` (1500x1000), `207.jpeg` (275x183), `170.jpeg` (275x183) | Equal resolution; larger file size (123,162 B vs 122,403 B) |
| **645** | NON CANCER | `val` | 3 | **`178.jpeg`** | 720x415 | 20,586 B | `197.jpeg` (720x415), `231.jpeg` (720x415) | Equal resolution & size; alphabetical filename tie-breaker |
| **648** | NON CANCER | `train` | 2 | **`182.jpeg`** | 498x508 | 33,670 B | `206.jpeg` (222x227) | Higher resolution (498x508 vs max 222x227) |
| **661** | NON CANCER | `train` | 2 | **`200.jpeg`** | 976x1000 | 113,064 B | `217.jpeg` (976x1000) | Equal resolution; larger file size (113,064 B vs 107,404 B) |
| **663** | NON CANCER | `val` | 2 | **`219.jpeg`** | 700x494 | 25,398 B | `202.jpeg` (612x431) | Higher resolution (700x494 vs max 612x431) |
