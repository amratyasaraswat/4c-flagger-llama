# CS 521 Project · Fine‑Tuning Llama‑3 8B on 4C‑Coded Clinical Transcripts

**One‑line pitch —** Build and evaluate an end‑to‑end pipeline that learns to detect _contextual red flags_, probes, factors, and contextualized care plans (“4C” framework) from physician‑patient transcripts by fine‑tuning an open‑source Llama‑3 8B model with PEFT/LoRA.

## 1 · Why this matters  
Clinical decisions fail when providers miss life‑context factors such as medication affordability or transportation barriers.  
The **4C content‑coding schema** highlights those signals, but manual annotation is expensive.  
This project investigates whether a medium‑sized instruction‑tuned model can:

1. **Extract 4C spans** (red flag → probe → factor → plan) from raw transcripts.  
2. **Generalize** across new encounters with minimal additional labeling.  
3. **Serve** as a component in real‑time clinical‑documentation tools.

---

## 2 · Dataset  

| Source | Size | License |
| --- | --- | --- |
| **Physician‑patient transcripts with 4C coding analysis** (VHA) | 405 de‑identified primary‑care transcripts | US Government public domain |

> **Get full data:**  
> In order to access the entire dataset, visit  
> <https://www.data.va.gov/dataset/Physician-patient-transcripts-with-4C-coding-analy/4qbs-wgct/about_data>

Small toy excerpts (2 transcripts + annotation) live in `data/fixtures/` so the repo remains lightweight.

---

## 3 · Repo layout  

```text
.
├── README.md
├── data/
│   ├──targets
│   ├──transcripts
├── docs
│   ├──prompt.txt
├── scripts/
│   ├── extract_4c_flags.py
│   ├── train_llama_4c.py
│   └── evaluate_4c_extraction.py
└── 4c_8b_lora
```

---

## 4 · Limitations 

This fine‑tuned model performs better than zero-shot Llama 8B model but it still is far from correctly flagging transcripts.
The main reason can be attributed to complexity for the model to learn red flags and limited data its being trained on (100 transcripts)

---

## 5 · Citation  

```bibtex
@misc{cs521_llama4c_2025,
  title   = {Fine‑Tuning Llama‑3 8B for 4C Clinical Context Extraction},
  author  = {Saraswat, Amratya},
  year    = {2025},
  howpublished = {\url{https://github.com/<user>/cs521-llama-4c}}
}
```

---

## 6 · License  

Code is released under the **MIT License** (`LICENSE`).  
Dataset is public domain via the U.S. Department of Veterans Affairs; review their page for usage guidance.

---

## 7 · Acknowledgements  

* Veterans Health Administration for releasing the 4C‑coded transcripts.  
* University of Illinois at Chicago **CS 521** faculty for project guidance.  
* Hugging Face community for open‑sourcing Llama‑3 weights and PEFT tools.

