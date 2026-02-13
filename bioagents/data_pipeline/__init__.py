"""Data pipeline for BIOAgents.

Handles:
- Raw dataset loading (MedQA, MedMCQA, MMLU, instruction data)
- Conversion to BIOAgents task format
- SFT training data generation from trajectories
- Evidence augmentation from knowledge base
"""
