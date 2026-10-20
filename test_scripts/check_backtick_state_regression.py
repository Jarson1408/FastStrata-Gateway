"""Regression test for backtick state tracking and turn boundary isolation.

When an assistant turn outputs a markdown table containing backticks
(e.g. `| Bloki ` ```dataview ` | 4179 |`), and then a new user turn arrives
(e.g. 'powtorz pytanie'), the gateway's parser for the new turn must NOT inherit
an open backtick state (in_triple=True or in_single=True).
"""
import unittest
from gateway.repair import check_text_backtick_state, get_initial_backtick_state


class TestBacktickRegression(unittest.TestCase):
    def test_markdown_table_with_dataview_inline_block(self):
        table_text = (
            "Znalezisko w 4290 plikach `.md`:\n\n"
            "| Element | Pliki |\n"
            "|---|---|\n"
            "| Bloki ` ```dataview ` | 4179 |\n"
            "| Wiki-linki `[[...]]` | 4205 |\n\n"
            "Zanim przetworzę 4290 plików, doprecyzuj zakres:\n\n"
            "<tool_call>\n<function=question>\n"
            "<parameter=questions>[]</parameter>\n"
            "</function>\n</tool_call>"
        )
        self.assertEqual(check_text_backtick_state(table_text), (False, False))

    def test_turn_boundary_isolation_prevents_backtick_poisoning(self):
        messages = [
            {"role": "assistant", "content": "| Bloki ` ```dataview ` | 4179 |"},
            {"role": "user", "content": "powtorz pytanie"},
        ]
        self.assertEqual(get_initial_backtick_state(messages), (False, False))

    def test_turn_boundary_isolation_with_open_triple(self):
        messages = [
            {"role": "assistant", "content": "```python\ndef foo():\n"},
            {"role": "user", "content": "stop and answer me"},
        ]
        self.assertEqual(get_initial_backtick_state(messages), (False, False))

    def test_assistant_prefill_retains_state(self):
        messages = [
            {"role": "user", "content": "code please"},
            {"role": "assistant", "content": "```python\ndef foo():\n"},
        ]
        self.assertEqual(get_initial_backtick_state(messages), (True, False))


if __name__ == "__main__":
    unittest.main()
