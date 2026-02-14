# SOAR: Confidence-Switched Position Beam Search for Diffusion Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.10953-b31b1b.svg)](https://arxiv.org/abs/2602.10953)

---

## 📖 Overview

<div align="center">
  <img src="figures/method.png" alt="Illustration of SOAR" width="1000">
</div>

SOAR is a confidence-switched position beam search decoding strategy for diffusion language models. The core idea is:

When there are high-confidence tokens in the sequence, SOAR selects parallel decoding for these tokens; otherwise, it employs position beam search to expand the search space.

---

## 🎯 Main Results

<p align="center">
  <img src="figures/decoding_strategy_impact.png" alt="Main Results" width="400">
</p>

SOAR achieves improved decoding quality without sacrificing decoding speed

---

## 🔧 Installation

```bash
git clone https://github.com/duterscmy/SOAR.git
cd SOAR
pip install transformers accelerate
```

## 📊 Evaluation

We evaluate SOAR using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI.

### For LLaDA-8B-Base
```bash
cd eval_llada8b
bash eval_soar_llada.sh
```

### For Dream-7B-Base:
```bash
cd eval_dream7b
bash eval_soar_dream.sh
```

## 📝 Citation
If you use SOAR in your research, please cite:

```
@misc{cao2026searchaccelerateconfidenceswitchedposition,
    title={Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models}, 
    author={Mingyu Cao and Alvaro Correia and Christos Louizos and Shiwei Liu and Lu Yin},
    year={2026},
    eprint={2602.10953},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2602.10953}, 
}
```


## Acknowledgments
This implementation is based on the [LLaDA](https://github.com/ML-GSAI/LLaDA) and [Dream](https://github.com/DreamLM/Dream) repositories. We thank the teams for open-sourcing their models and code.