# online-learning
Python framework for online learning in machine learning with streaming data pipelines, concept drift detection, and prequential evaluation.

# Online Learning in Python: How to Train Models on Streaming Data

Complete code for **Article 06** of the [Production ML Engineering series](https://emitechlogic.com/machine-learning-production-pipeline/) at EmiTechLogic.

> **Series:** Production ML Engineering — [Article 06 of 15](https://emitechlogic.com/online-learning-machine-learning-python/) (Cluster 2: Continual Learning)  
> **Previous:** [Catastrophic Forgetting in PyTorch (Article 05)](https://emitechlogic.com/how-to-prevent-catastrophic-forgetting-in-pytorch/)  
> **Next:** Continual Learning in PyTorch (Article 07)

---

## What This Code Covers

| Article Section | Module |
|---|---|
| Batch vs Online Learning | `data/generators.py` — stream vs batch data source design |
| Setting Up a Streaming Data Pipeline | `pipelines/streaming_pipeline.py` |
| Updating Weights on Each Sample | `methods/sgd_online.py` (PyTorch SGD) |
| Online Learning Algorithms: SGD, River | `methods/river_learner.py` |
| How to Handle Concept Drift | `methods/drift_detector.py` |
| Evaluation Without a Held-Out Set | `evaluation/prequential.py` |
| Fraud Detection Use Case | `use_cases/fraud_detection.py` |
| Recommendation Systems Use Case | `use_cases/recommendation.py` |
| Head-to-Head Benchmark | `benchmarks/benchmark.py` |

---

## Repository Structure

```
online-learning/
├── data/
│   └── generators.py          # SEA, Hyperplane, Fraud, Recommendation streams
├── methods/
│   ├── sgd_online.py          # PyTorch one-sample-at-a-time SGD learners
│   ├── river_learner.py       # River: LogisticRegression, HoeffdingTree, AdaptiveRF
│   └── drift_detector.py      # ADWIN, DDM, Page-Hinkley drift detectors
├── evaluation/
│   └── prequential.py         # Test-then-train evaluator + OnlineMetrics
├── pipelines/
│   └── streaming_pipeline.py  # Production streaming pipeline class
├── use_cases/
│   ├── fraud_detection.py     # Fraud detection with drift response
│   └── recommendation.py      # Online CTR prediction
├── benchmarks/
│   └── benchmark.py           # Head-to-head: Batch vs SGD vs River vs ARF
└── tests/
    └── test_online_learning.py
```

---

## Quickstart

```bash
git clone https://github.com/Emmimal/online-learning
cd online-learning
pip install -r requirements.txt
```

### Run the benchmark

```bash
python benchmarks/benchmark.py
```

### Run the fraud detection use case

```bash
python use_cases/fraud_detection.py
```

### Run the recommendation use case

```bash
python use_cases/recommendation.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Core Concept: Prequential Evaluation

The prequential loop is used throughout this codebase. The order is non-negotiable:

```python
for x, y in stream:
    y_pred = model.predict_one(x)   # 1. Predict BEFORE learning
    evaluate(y, y_pred)             # 2. Record metric
    detector.update(y, y_pred)      # 3. Check for drift
    model.learn_one(x, y)           # 4. Learn AFTER predicting
```

Swapping steps 1 and 4 gives the model access to the label before it predicts — producing optimistic accuracy estimates that do not reflect real deployment performance.

---

## Minimal Working Example

```python
from data.generators import SEAConceptStream
from methods.river_learner import RiverAdaptiveRF
from methods.drift_detector import ADWIN
from pipelines.streaming_pipeline import StreamingPipeline, DriftResponse

pipeline = StreamingPipeline(
    model=RiverAdaptiveRF(n_models=10),
    drift_detector=ADWIN(delta=0.002),
    drift_response=DriftResponse.log_only(),
    window_size=500,
    verbose=True,
)

stream = SEAConceptStream(n_samples=10_000, drift_at=5_000)
result = pipeline.run(stream)
print(result.summary())
```

---

## Environment

Tested on:
- Python 3.12
- PyTorch 2.0+
- River 0.21+
- Windows 10 / Ubuntu 22.04

---

## Disclosure

**Code authorship:** All code in this repository is the original work of the author.  
**Benchmark authenticity:** All benchmark numbers in the article are from real runs.  
**No affiliate relationships:** All tools mentioned are open-source under MIT or BSD licenses.  
**Series affiliation:** Article 06 of the Production ML Engineering series at [EmiTechLogic](https://emitechlogic.com).
