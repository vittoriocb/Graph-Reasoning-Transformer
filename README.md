# The Graph Reasoning Transformer (GRT)

This repository contains the original article for the **Graph Reasoning Transformer (GRT)**, a novel architecture proposed by Vittorio Calcagno.

The GRT is a theoretical proposal for achieving true, verifiable machine reasoning, representing a fundamental paradigm shift away from the associative intelligence of contemporary Large Language Models (LLMs).

### The Core Problem

Current frontier models are masters of mimicry, not deduction. Their intelligence is based on recognizing statistical patterns in text, not on performing logical operations. This results in an unreliable illusion of logic, marked by hallucinations and inconsistency when faced with novel problems.

### Our Proposed Solution: A New Paradigm

This architecture is built on a foundational shift in the computational primitive: from a system that operates on **`Words`** to produce **`Associations`**, to a system that operates on **`Knowledge`** to produce new **`Reasoning`**.

The GRT is a small, efficient, decoder-only transformer designed to achieve this. Its key innovations are:

* **Knowledge as the Primitive:** The model operates on structured **Subject-Verb-Object (SVO) fact vectors**, not ambiguous text tokens.
* **Reasoning as an Interaction-Product:** The central mechanism is a novel QKRV-Attention where the logical relationship (`R`) between two facts is not retrieved, but **computed dynamically as the interaction-product of a query fact (`Q`) and a key fact (`K`)**. The edge between facts *is* the reason.
* **The "Honest Student" Bootstrap:** We leverage a large LLM (e.g., GPT-4, Claude 4, Gemini 2.5) for a one-time knowledge extraction task, then use that corpus to train a **small, computationally simple model that is forced to learn true logic** rather than relying on memorization.
* **A Path to AGI:** The small model's blazing-fast inference speed is the critical accelerator for a **recursive self-improvement loop**. By reasoning about its own architecture, the system can improve itself exponentially, creating a direct path to AGI.

The full article can be read in `The_Graph_Reasoning_Transformer.pdf`.
