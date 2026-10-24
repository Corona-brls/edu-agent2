from __future__ import annotations

from backend.resources.portal import _present_item


def test_summary_selects_topic_relevant_later_sentence() -> None:
    irrelevant = "This opening sentence is unrelated to the requested course topic. " * 8
    item = {
        "type": "paper",
        "title": "5G NAS authentication study",
        "source_name": "arxiv",
        "module": "5G NAS",
        "tags": ["5G NAS", "security"],
        "raw_summary": (
            irrelevant
            + "The paper evaluates 5G NAS authentication replay resistance and key agreement."
        ),
        "url": "https://example.test/paper",
    }

    presented = _present_item(item)

    assert "authentication replay" in presented["summary"]
    assert presented["topic_category"] == "5G NAS"
    assert presented["raw_summary"].startswith("This opening sentence")


def test_unknown_topic_is_not_forced_into_a_known_category() -> None:
    presented = _present_item(
        {
            "type": "paper",
            "title": "Acoustic signal classification",
            "source_name": "arxiv",
            "raw_summary": "A study of acoustic signal classification.",
            "tags": ["acoustics"],
        }
    )

    assert presented["topic_category"] == "\u5f85\u5206\u7c7b"


def test_standard_summary_is_metadata_only_without_fabricating_full_text() -> None:
    presented = _present_item(
        {
            "type": "standard",
            "title": "3GPP TS 24.501",
            "source_name": "standard_source",
            "raw_summary": "Version 17.4; official metadata link only.",
            "url": "https://example.test/ts-24-501",
            "year": None,
            "metadata": {
                "metadata_only": True,
                "local_full_text_available": False,
            },
        }
    )

    assert presented["metadata"]["metadata_only"] is True
    assert presented["metadata"]["local_full_text_available"] is False
    assert "version 17.4" in presented["summary"].lower()
    assert "normative clause 1" not in presented["summary"].lower()
