"""Medical tools for the Medical QA domain.

Provides tools for:
- PubMed-style literature search
- Article browsing
- Medical wiki / encyclopedia search
- Evidence retrieval (MedCPT-style)
- Option analysis helper
- Internal reasoning (think)
"""

import re
from typing import List, Optional

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool
from bioagents.domains.medical_qa.data_model import (
    MedicalQADB,
    Article,
    EvidencePassage,
    WikiEntry,
    SearchLog,
)


def _simple_relevance(query: str, text: str) -> float:
    """Compute a simple keyword-overlap relevance score (0-1)."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    text_tokens = set(re.findall(r"\w+", text.lower()))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / len(query_tokens)


class MedicalQATools(ToolKitBase):
    """Tools available to the medical QA agent."""

    db: MedicalQADB

    def __init__(self, db: MedicalQADB) -> None:
        super().__init__(db)

    # ==========================================
    # Category 1: Literature Search (PubMed-style)
    # ==========================================

    @is_tool(ToolType.READ)
    def search_pubmed(self, query: str, max_results: int = 5) -> list:
        """Search PubMed-style medical literature for articles relevant to the query.

        Args:
            query: Search query (e.g., 'cisplatin ototoxicity mechanism')
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of matching articles with title, abstract snippet, and PMID
        """
        max_results = int(max_results)
        query_lower = query.lower()

        scored = []
        for pmid, article in self.db.articles.items():
            combined = f"{article.title} {article.abstract} {' '.join(article.keywords)}"
            score = _simple_relevance(query, combined)
            if score > 0:
                scored.append((score, pmid, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, pmid, article in scored[:max_results]:
            abstract_snippet = article.abstract[:300]
            if len(article.abstract) > 300:
                abstract_snippet += "..."
            results.append({
                "pmid": pmid,
                "title": article.title,
                "abstract_snippet": abstract_snippet,
                "journal": article.journal,
                "year": article.year,
                "relevance": round(score, 3),
            })

        # Log the search
        self.db.search_log.append(
            SearchLog(query=query, results_count=len(results))
        )

        if not results:
            return [{
                "message": f"No articles found for '{query}'. Try broader search terms.",
                "suggestion": "Consider using medical terminology or disease names.",
            }]
        return results

    @is_tool(ToolType.READ)
    def browse_article(self, pmid: str, section: str = "") -> dict:
        """Browse a specific article by PMID. Optionally read a specific section.

        Args:
            pmid: The PubMed ID of the article to browse
            section: Optional section heading to read (e.g., 'methods', 'results'). Empty returns full abstract.

        Returns:
            Article details including title, full abstract, and sections if available
        """
        if pmid not in self.db.articles:
            return {"error": f"Article with PMID '{pmid}' not found."}

        article = self.db.articles[pmid]
        result = {
            "pmid": article.pmid,
            "title": article.title,
            "authors": article.authors,
            "journal": article.journal,
            "year": article.year,
            "keywords": article.keywords,
        }

        if section and section.lower() in {k.lower(): k for k in article.sections}:
            # Find the matching section (case-insensitive)
            matched_key = next(
                k for k in article.sections if k.lower() == section.lower()
            )
            result["section"] = matched_key
            result["content"] = article.sections[matched_key]
        else:
            result["abstract"] = article.abstract
            if article.sections:
                result["available_sections"] = list(article.sections.keys())

        return result

    # ==========================================
    # Category 2: Medical Wiki / Encyclopedia
    # ==========================================

    @is_tool(ToolType.READ)
    def search_medical_wiki(self, query: str, max_results: int = 5) -> list:
        """Search the medical encyclopedia / wiki for entries related to the query.

        Args:
            query: Search query (e.g., 'sensorineural hearing loss')
            max_results: Maximum number of results to return (default 5)

        Returns:
            List of matching encyclopedia entries with title and summary snippet
        """
        max_results = int(max_results)

        scored = []
        for eid, entry in self.db.wiki_entries.items():
            combined = f"{entry.title} {entry.summary} {' '.join(entry.categories)}"
            score = _simple_relevance(query, combined)
            if score > 0:
                scored.append((score, eid, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, eid, entry in scored[:max_results]:
            summary_snippet = entry.summary[:250]
            if len(entry.summary) > 250:
                summary_snippet += "..."
            results.append({
                "entry_id": eid,
                "title": entry.title,
                "summary": summary_snippet,
                "categories": entry.categories,
                "relevance": round(score, 3),
            })

        if not results:
            return [{
                "message": f"No encyclopedia entries found for '{query}'.",
                "suggestion": "Try alternative medical terms or disease names.",
            }]
        return results

    @is_tool(ToolType.READ)
    def browse_wiki_entry(self, entry_id: str) -> dict:
        """Browse a specific medical encyclopedia entry by its ID.

        Args:
            entry_id: The unique identifier of the wiki entry

        Returns:
            Full entry content including title, summary, categories, and full text
        """
        if entry_id not in self.db.wiki_entries:
            return {"error": f"Entry '{entry_id}' not found."}

        entry = self.db.wiki_entries[entry_id]
        result = {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "summary": entry.summary,
            "categories": entry.categories,
        }
        if entry.full_text:
            result["full_text"] = entry.full_text
        if entry.related_entries:
            result["related_entries"] = entry.related_entries
        return result

    # ==========================================
    # Category 3: Evidence Retrieval (MedCPT-style)
    # ==========================================

    @is_tool(ToolType.READ)
    def retrieve_evidence(self, query: str, max_results: int = 5, category: str = "") -> list:
        """Retrieve relevant evidence passages from medical textbooks and literature.

        Args:
            query: The medical question or topic to find evidence for
            max_results: Maximum number of passages to return (default 5)
            category: Optional category filter (e.g., 'pharmacology', 'pathology', 'anatomy')

        Returns:
            List of evidence passages ranked by relevance
        """
        max_results = int(max_results)

        candidates = self.db.evidence_passages.values()
        if category:
            candidates = [
                p for p in candidates if p.category.lower() == category.lower()
            ]

        scored = []
        for passage in candidates:
            combined = f"{passage.title} {passage.text}"
            score = _simple_relevance(query, combined)
            if score > 0:
                scored.append((score, passage))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, passage in scored[:max_results]:
            text_snippet = passage.text[:400]
            if len(passage.text) > 400:
                text_snippet += "..."
            results.append({
                "passage_id": passage.passage_id,
                "source": passage.source,
                "title": passage.title,
                "text": text_snippet,
                "category": passage.category,
                "relevance": round(score, 3),
            })

        if not results:
            return [{
                "message": f"No evidence found for '{query}'.",
                "suggestion": "Try different medical terms or broaden the query.",
            }]
        return results

    # ==========================================
    # Category 4: Answer Analysis
    # ==========================================

    @is_tool(ToolType.READ)
    def analyze_answer_options(self, question: str, options: str) -> dict:
        """Analyze the given answer options against the knowledge base to find supporting evidence.

        Args:
            question: The medical question text
            options: Comma-separated answer options (e.g., 'A: Inhibition of proteasome, B: Cross-linking of DNA')

        Returns:
            Analysis of each option with supporting/contradicting evidence from the knowledge base
        """
        # Parse options
        option_list = [o.strip() for o in options.split(",") if o.strip()]

        analysis = {}
        for opt in option_list:
            # Try to split label:text
            parts = opt.split(":", 1)
            label = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else opt

            # Find supporting evidence
            combined_query = f"{question} {text}"
            scored = []
            for passage in self.db.evidence_passages.values():
                combined = f"{passage.title} {passage.text}"
                score = _simple_relevance(combined_query, combined)
                if score > 0.1:
                    scored.append((score, passage))

            scored.sort(key=lambda x: x[0], reverse=True)
            evidence_snippets = []
            for s, p in scored[:2]:
                evidence_snippets.append({
                    "source": p.source,
                    "text": p.text[:200],
                    "relevance": round(s, 3),
                })

            analysis[label] = {
                "option_text": text,
                "supporting_evidence_count": len(scored),
                "evidence": evidence_snippets,
            }

        return {
            "question_summary": question[:200],
            "options_analysis": analysis,
            "note": "Use this analysis along with your medical knowledge to select the best answer.",
        }

    # ==========================================
    # Category 5: Reasoning / Think
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use this to think through complex medical questions before answering.

        Args:
            thought: Your reasoning process for the medical question

        Returns:
            Empty string (thinking is logged but produces no output)
        """
        return ""

    @is_tool(ToolType.GENERIC)
    def submit_answer(self, answer: str, reasoning: str = "") -> str:
        """Submit your final answer to the medical question.

        Args:
            answer: The answer label (e.g., 'A', 'B', 'C', 'D')
            reasoning: Your reasoning for selecting this answer

        Returns:
            Confirmation of the submitted answer
        """
        return f"Answer '{answer}' submitted. Reasoning: {reasoning}"

    # ==========================================
    # Assertion helpers (for evaluation)
    # ==========================================

    def assert_correct_answer(self, question_id: str, submitted_answer: str) -> bool:
        """Check if the submitted answer is correct."""
        if question_id not in self.db.questions:
            return False
        question = self.db.questions[question_id]
        return submitted_answer.strip().upper() == question.correct_answer.strip().upper()

    def assert_evidence_retrieved(self, question_id: str, search_log: list) -> bool:
        """Check if the agent searched for relevant evidence."""
        return len(search_log) > 0 and any(
            s.results_count > 0 for s in self.db.search_log
        )
