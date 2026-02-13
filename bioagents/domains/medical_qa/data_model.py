"""Data models for the Medical QA domain.

Defines the knowledge base schema including:
- Medical articles (PubMed-style)
- Evidence passages (for retrieval)
- Medical encyclopedia entries (wiki-style)
- QA questions with multiple choice options
"""

import os
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from bioagents.environment.db import DB


# --- Sub-models ---


class Article(BaseModel):
    """A medical journal article (PubMed-style)."""
    pmid: str = Field(description="PubMed ID")
    title: str = Field(description="Article title")
    abstract: str = Field(description="Article abstract")
    authors: List[str] = Field(default_factory=list, description="Author list")
    journal: str = Field(default="", description="Journal name")
    year: int = Field(default=2024, description="Publication year")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    doi: str = Field(default="", description="DOI")
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional full-text sections {heading: content}",
    )


class EvidencePassage(BaseModel):
    """A passage from a medical text used as evidence."""
    passage_id: str = Field(description="Unique passage identifier")
    source: str = Field(description="Source document or textbook")
    title: str = Field(default="", description="Section title")
    text: str = Field(description="Passage content")
    relevance_score: float = Field(default=0.0, description="Relevance to a query")
    category: str = Field(
        default="general",
        description="Category (e.g., 'pharmacology', 'pathology', 'anatomy')",
    )


class WikiEntry(BaseModel):
    """A medical encyclopedia / wiki entry."""
    entry_id: str = Field(description="Unique entry identifier")
    title: str = Field(description="Entry title")
    url: str = Field(default="", description="URL")
    summary: str = Field(description="Short summary / first paragraph")
    full_text: str = Field(default="", description="Full article text")
    categories: List[str] = Field(default_factory=list, description="Categories")
    related_entries: List[str] = Field(
        default_factory=list, description="IDs of related entries"
    )


class AnswerOption(BaseModel):
    """A single answer option for a multiple-choice question."""
    label: str = Field(description="Option label (A, B, C, D, E)")
    text: str = Field(description="Option text")


class MedicalQuestion(BaseModel):
    """A medical multiple-choice question."""
    question_id: str = Field(description="Unique question identifier")
    source: str = Field(
        default="MedQA",
        description="Source dataset (MedQA, MedMCQA, MMLU, custom)",
    )
    question: str = Field(description="The question text")
    options: List[AnswerOption] = Field(description="Answer options")
    correct_answer: str = Field(description="Correct answer label (A/B/C/D)")
    explanation: str = Field(default="", description="Explanation for the correct answer")
    category: str = Field(default="general", description="Medical category/topic")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium")
    relevant_evidence_ids: List[str] = Field(
        default_factory=list,
        description="IDs of evidence passages relevant to this question",
    )


class SearchLog(BaseModel):
    """Log entry for a search action."""
    query: str = Field(description="Search query")
    results_count: int = Field(default=0, description="Number of results returned")
    timestamp: str = Field(default="", description="Time of search")


# --- Main Database ---


class MedicalQADB(DB):
    """Medical QA domain database.

    Contains articles, evidence passages, wiki entries, and questions
    for the medical question-answering simulation.
    """

    articles: Dict[str, Article] = Field(
        default_factory=dict,
        description="PubMed-style articles indexed by PMID",
    )
    evidence_passages: Dict[str, EvidencePassage] = Field(
        default_factory=dict,
        description="Evidence passages indexed by passage_id",
    )
    wiki_entries: Dict[str, WikiEntry] = Field(
        default_factory=dict,
        description="Wiki entries indexed by entry_id",
    )
    questions: Dict[str, MedicalQuestion] = Field(
        default_factory=dict,
        description="Question bank indexed by question_id",
    )
    search_log: List[SearchLog] = Field(
        default_factory=list,
        description="Log of searches made during interaction",
    )


# --- Data paths ---

_DOMAIN_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "data", "domains", "medical_qa",
)
DB_PATH = os.path.join(_DOMAIN_DATA_DIR, "db.json")
POLICY_PATH = os.path.join(_DOMAIN_DATA_DIR, "policy.md")
TASKS_PATH = os.path.join(_DOMAIN_DATA_DIR, "tasks.json")


def get_db() -> MedicalQADB:
    """Load the medical QA database."""
    return MedicalQADB.load(DB_PATH)
