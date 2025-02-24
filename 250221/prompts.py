# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""A file containing prompts definition."""

from __future__ import annotations

GRAPH_EXTRACTION_PROMPT = """
Extract the headers from the given Markdown table. Return only the headers as a comma-separated list without any additional text or explanations.

Table Input:

{table_input}

Output: [List of headers]
"""

CONTINUE_PROMPT = "SOME other entities and relationship were missed in the last extraction.  Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities may have still been missed.  Answer {tuple_delimiter}YES{tuple_delimiter} if there are still entities that need to be added else {tuple_delimiter}NO{tuple_delimiter} \n"
