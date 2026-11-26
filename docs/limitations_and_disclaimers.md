# ORALSCAN AI: Research Limitations and Scientific Disclaimers

**Project:** Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI  
**Subject:** Methodological Deviations, Dataset Constraints, Interpretability Boundaries, and Regulatory Notices  
**Date:** November 2026  

---

## 1. Academic Research Purpose & Non-Diagnostic Notice

**ORALSCAN AI IS AN EXPERIMENTAL ACADEMIC RESEARCH PROTOTYPE.**  
It was created solely as part of a final-year engineering and computer science research project to evaluate the technical feasibility and interpretability of hybrid Vision Transformers (ViT) and Graph Attention Networks (GATv2) for photographic oral screening.

### Absolute Prohibitions:
- The system is **NOT a Medical Device** and has not been cleared, certified, or approved by the FDA, CE, CDSCO, or any medical regulatory authority.
- The system must **NEVER be used as a standalone diagnostic tool**, clinical decision support system, or substitute for physical clinical inspection.
- The system must **NEVER be used to confirm that a patient is cancer-free** or to definitively diagnose oral squamous cell carcinoma (OSCC).

---

## 2. Dataset and Epidemiological Constraints

1. **Representative Images vs Independent Patients:**
   The primary dataset used across Phases 1 through 6 consists of **703 representative images** (494 Train, 108 Validation, 101 Test in approved primary split; or 703 total de-duplicated images). The public dataset lacks patient-level identifiers; therefore, **patient-level independence cannot be certified**. A single patient may have multiple photographs from different angles or visits.
2. **Internal Reused Test Split:**
   The internal test set ($N=101$ to $109$ images) was evaluated across multiple experimental iterations during Phases 1–6. It represents an internal validation partition, NOT an untouched external cohort collected from independent clinical centers.
3. **Absence of Clinical Metadata:**
   The neural network evaluates only $224 \times 224$ photographic RGB pixel grids. The model **cannot** determine patient age, biological sex, smoking history, tobacco/areca nut chewing habits, alcohol intake, HPV status, lesion duration, pain symptoms, or medical comorbidities. Any claim that the system evaluates patient lifestyle or clinical history is medically false.

---

## 3. Methodological Deviations & Historical Experiments

1. **Phase 5–6 Class-Weight Configuration:**
   During historical training in Phase 5 and Phase 6, a class-weighting configuration was applied that differed slightly from the exact mathematical inverse class frequencies of the training split. This historical parameter is documented as an experimental variation; the model weights are frozen and certified.
2. **Comparative Handcrafted Features vs Hybrid ViT-GAT:**
   In Phase 6, an auxiliary comparative classifier using handcrafted texture (LBP, GLCM), color (HSV/Lab), and edge (Sobel) features achieved 87.16% test accuracy, while the hybrid ViT-GATv2 achieved 77.06%. Handcrafted features represent mathematical image statistics; they are **NOT clinically validated biomarkers**. The central objective of this research is deep learning interpretability using ViT-GAT architectures, and the application demonstrates the deep hybrid pipeline honestly without suppressing the comparative baseline results.

---

## 4. Explainable AI (XAI) Methodological Boundaries

### What the Visual Explanations Show:
- **ViT Attention Rollout:** Shows which spatial regions had high attention weights feeding into the class token $[\text{CLS}]$.
- **GATv2 Overlays:** Shows inter-patch graph connectivity and topological message passing.
- **Integrated Gradients:** Quantifies the relative first-order gradient sensitivity flowing from the visual versus graph branches into the fusion head along a linear interpolation path from reference baseline $[0.05]_{384}$.

### What the Visual Explanations Do NOT Show:
- **No Surgical Margins:** Saliency heatmaps do NOT delineate surgical resection borders or biological tumor depth.
- **No Cellular Dysplasia:** Elevated visual attention indicates model feature sensitivity, not the presence of dysplastic cells, hyperkeratosis, or carcinoma in situ.
- **No Causal Proof:** Attention weights indicate correlation and feature routing in an artificial neural network; they do not prove biological cause of disease.

---

## 5. Mandatory Clinical Referral Guidance

Any oral mucosal red lesion (erythroplakia), white lesion (leukoplakia), mixed lesion (erythroleukoplakia), non-healing ulcer, or tissue mass persisting for **more than 14 days** requires mandatory physical examination by a qualified dental surgeon, maxillofacial specialist, or head-and-neck oncologist.

**Gold Standard:** Surgical incisional tissue biopsy accompanied by histopathological microscopic evaluation remains the **sole definitive medical gold standard** for diagnosing oral cancer.
