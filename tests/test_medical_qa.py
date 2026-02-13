"""Tests for the Medical QA domain."""

import json
import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMedicalQADB:
    """Test Medical QA database loading and structure."""

    def test_load_db(self):
        from bioagents.domains.medical_qa.data_model import MedicalQADB, DB_PATH
        db = MedicalQADB.load(DB_PATH)
        assert len(db.articles) > 0, "Should have articles"
        assert len(db.evidence_passages) > 0, "Should have evidence passages"
        assert len(db.wiki_entries) > 0, "Should have wiki entries"

    def test_article_structure(self):
        from bioagents.domains.medical_qa.data_model import MedicalQADB, DB_PATH
        db = MedicalQADB.load(DB_PATH)
        for pmid, article in db.articles.items():
            assert article.pmid == pmid
            assert article.title
            assert article.abstract

    def test_evidence_structure(self):
        from bioagents.domains.medical_qa.data_model import MedicalQADB, DB_PATH
        db = MedicalQADB.load(DB_PATH)
        for pid, passage in db.evidence_passages.items():
            assert passage.passage_id == pid
            assert passage.text
            assert passage.source


class TestMedicalQATools:
    """Test Medical QA tools."""

    @pytest.fixture
    def tools(self):
        from bioagents.domains.medical_qa.data_model import MedicalQADB, DB_PATH
        from bioagents.domains.medical_qa.tools import MedicalQATools
        db = MedicalQADB.load(DB_PATH)
        return MedicalQATools(db)

    def test_tools_registered(self, tools):
        all_tools = tools.tools
        assert "search_pubmed" in all_tools
        assert "browse_article" in all_tools
        assert "search_medical_wiki" in all_tools
        assert "browse_wiki_entry" in all_tools
        assert "retrieve_evidence" in all_tools
        assert "analyze_answer_options" in all_tools
        assert "think" in all_tools
        assert "submit_answer" in all_tools
        assert len(all_tools) >= 8

    def test_search_pubmed(self, tools):
        results = tools.search_pubmed("cisplatin ototoxicity")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "pmid" in results[0] or "message" in results[0]

    def test_search_pubmed_no_results(self, tools):
        results = tools.search_pubmed("xyznotexist12345")
        assert isinstance(results, list)
        assert "message" in results[0]

    def test_browse_article(self, tools):
        result = tools.browse_article("PMID001")
        assert "title" in result
        assert "Cisplatin" in result["title"]

    def test_browse_article_not_found(self, tools):
        result = tools.browse_article("INVALID")
        assert "error" in result

    def test_browse_article_section(self, tools):
        result = tools.browse_article("PMID001", section="mechanism")
        assert "content" in result
        assert "DNA cross-links" in result["content"]

    def test_search_medical_wiki(self, tools):
        results = tools.search_medical_wiki("allergic conjunctivitis")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_browse_wiki_entry(self, tools):
        result = tools.browse_wiki_entry("WIKI001")
        assert "title" in result
        assert "Cisplatin" in result["title"]

    def test_retrieve_evidence(self, tools):
        results = tools.retrieve_evidence("cisplatin DNA cross-linking")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "text" in results[0]

    def test_retrieve_evidence_with_category(self, tools):
        results = tools.retrieve_evidence(
            "cisplatin mechanism", category="pharmacology"
        )
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["category"] == "pharmacology"

    def test_analyze_answer_options(self, tools):
        result = tools.analyze_answer_options(
            question="What is the mechanism of cisplatin?",
            options="A: Proteasome inhibition, B: DNA cross-linking",
        )
        assert "options_analysis" in result

    def test_think(self, tools):
        result = tools.think("Let me consider the pharmacology...")
        assert result == ""

    def test_submit_answer(self, tools):
        result = tools.submit_answer("D", "Cross-linking of DNA")
        assert "D" in result
        assert "submitted" in result.lower()


class TestMedicalQAEnvironment:
    """Test Medical QA environment setup."""

    def test_get_environment(self):
        from bioagents.domains.medical_qa.environment import get_environment
        env = get_environment()
        assert env.domain_name == "medical_qa"
        assert env.tools is not None
        assert len(env.tools.tools) >= 8

    def test_get_tasks(self):
        from bioagents.domains.medical_qa.environment import get_tasks
        tasks = get_tasks()
        assert len(tasks) >= 10, f"Expected at least 10 tasks, got {len(tasks)}"
        for task in tasks:
            assert "id" in task
            assert "ticket" in task
            assert "correct_answer" in task
            assert "evaluation_criteria" in task

    def test_environment_step(self):
        from bioagents.domains.medical_qa.environment import get_environment
        env = get_environment()
        env.reset()

        # Execute a search tool call
        action = json.dumps({
            "name": "search_pubmed",
            "arguments": {"query": "cisplatin ototoxicity"},
        })
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs  # Should have search results
        assert not terminated


class TestMedicalQAGym:
    """Test Gymnasium interface for Medical QA."""

    def test_gym_env_create(self):
        from bioagents.gym.agent_env import BioAgentGymEnv
        env = BioAgentGymEnv(domain="medical_qa")
        assert env.domain_name == "medical_qa"
        assert len(env._tasks) >= 10

    def test_gym_env_reset(self):
        from bioagents.gym.agent_env import BioAgentGymEnv
        env = BioAgentGymEnv(domain="medical_qa")
        obs, info = env.reset(options={"task_id": "medqa_cisplatin_001"})
        assert "cisplatin" in obs.lower() or "bladder" in obs.lower()
        assert "tools" in info
        assert info["domain"] == "medical_qa"

    def test_gym_env_full_episode(self):
        from bioagents.gym.agent_env import BioAgentGymEnv
        env = BioAgentGymEnv(domain="medical_qa", max_turns=10)
        obs, info = env.reset(options={"task_id": "medqa_sids_001"})

        # Step 1: Search for evidence
        action1 = json.dumps({
            "name": "search_medical_wiki",
            "arguments": {"query": "SIDS prevention supine sleeping"},
        })
        obs1, reward1, term1, trunc1, info1 = env.step(action1)
        assert not term1
        assert reward1 == 0.0  # No reward during interaction

        # Step 2: Submit answer
        action2 = json.dumps({
            "name": "submit_answer",
            "arguments": {"answer": "A", "reasoning": "Supine sleeping on firm mattress prevents SIDS"},
        })
        obs2, reward2, term2, trunc2, info2 = env.step(action2)
        # Episode should still be running (not terminated by submit_answer alone)

        # Get trajectory
        trajectory = env.get_trajectory()
        assert trajectory["domain"] == "medical_qa"
        assert trajectory["total_turns"] >= 2
        assert len(trajectory["tool_call_log"]) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
