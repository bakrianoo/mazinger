"""The description stage must survive whatever shape the LLM returns.

``json_repair`` fixes malformed *syntax*; it does not make a model honour the
requested *schema*. Smaller local models (the Ollama default, for one) routinely
wrap list entries in objects. Because ``describe`` runs after download,
transcription and thumbnail extraction, a crash here throws away several minutes
of GPU work — so every one of these shapes must degrade rather than raise.
"""

import pytest

from mazinger.describe import _as_text, _dedupe_texts


class TestAsText:
    def test_plain_string_is_stripped(self):
        assert _as_text("  fountain pens  ") == "fountain pens"

    @pytest.mark.parametrize(
        "item",
        [
            {"keypoint": "nib flexibility"},
            {"keyword": "nib flexibility"},
            {"point": "nib flexibility"},
            {"text": "nib flexibility"},
            {"name": "nib flexibility"},
            {"term": "nib flexibility"},
            {"value": "nib flexibility"},
        ],
        ids=lambda d: next(iter(d)),
    )
    def test_conventional_wrapper_keys_are_unwrapped(self, item):
        assert _as_text(item) == "nib flexibility"

    def test_unknown_key_falls_back_to_first_string_value(self):
        assert _as_text({"unexpected": "nib flexibility"}) == "nib flexibility"

    def test_preferred_key_wins_over_other_values(self):
        item = {"why": "because it matters", "keypoint": "nib flexibility"}
        assert _as_text(item) == "nib flexibility"

    def test_dict_without_any_string_value_yields_empty(self):
        assert _as_text({"count": 3, "ok": True}) == ""

    def test_nested_list_is_flattened(self):
        assert _as_text(["ink", "nib"]) == "ink nib"

    @pytest.mark.parametrize(("item", "expected"), [(None, ""), (42, "42"), (True, "True")])
    def test_scalars_and_none(self, item, expected):
        assert _as_text(item) == expected


class TestDedupeTexts:
    def test_deduplicates_case_insensitively_and_preserves_order(self):
        assert _dedupe_texts(["Ink", "nib", "INK", "Nib "], 10) == ["Ink", "nib"]

    def test_drops_empty_entries(self):
        assert _dedupe_texts(["ink", "", "   ", None, {}], 10) == ["ink"]

    def test_respects_the_limit(self):
        assert _dedupe_texts([f"kw{i}" for i in range(50)], 20) == [f"kw{i}" for i in range(20)]

    def test_mixed_strings_and_objects(self):
        items = ["ink", {"keyword": "nib"}, {"term": "ink"}, 7]
        assert _dedupe_texts(items, 10) == ["ink", "nib", "7"]

    @pytest.mark.parametrize("items", ["not a list", {"a": 1}, None, 5])
    def test_non_list_input_yields_empty_list(self, items):
        assert _dedupe_texts(items, 10) == []


class TestDescribeContentResilience:
    """End-to-end through ``describe_content`` with a stubbed LLM client."""

    @staticmethod
    def _client(payload):
        class _Msg:
            content = payload

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            def create(self, **_kwargs):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        return _Client()

    def _describe(self, payload):
        from mazinger.describe import describe_content

        return describe_content("1\n00:00:00,000 --> 00:00:01,000\nhi\n", [],
                                self._client(payload), llm_model="stub")

    def test_objects_inside_keypoints_do_not_raise(self):
        """The exact shape that killed a real run mid-pipeline."""
        out = self._describe(
            '{"title": "Pens", "summary": "About pens.",'
            ' "keypoints": [{"keypoint": "flexible nib"}, {"keypoint": "ink flow"}],'
            ' "keywords": [{"term": "nib"}, "ink"]}'
        )
        assert out["keypoints"] == ["flexible nib", "ink flow"]
        assert out["keywords"] == ["nib", "ink"]

    def test_a_bare_list_response_degrades_to_an_empty_description(self):
        out = self._describe('["not", "an", "object"]')
        assert out == {"title": "", "summary": "", "keypoints": [], "keywords": []}

    def test_a_plain_string_response_degrades_to_an_empty_description(self):
        out = self._describe('"I could not analyse this video."')
        assert out["title"] == ""
        assert out["keypoints"] == []

    def test_well_formed_response_is_unchanged(self):
        out = self._describe(
            '{"title": "Pens", "summary": "About pens.",'
            ' "keypoints": ["flexible nib"], "keywords": ["nib", "ink"]}'
        )
        assert out["title"] == "Pens"
        assert out["keypoints"] == ["flexible nib"]
        assert out["keywords"] == ["nib", "ink"]

    def test_missing_list_fields_are_left_absent(self):
        out = self._describe('{"title": "Pens", "summary": "About pens."}')
        assert out["title"] == "Pens"
        assert "keypoints" not in out
