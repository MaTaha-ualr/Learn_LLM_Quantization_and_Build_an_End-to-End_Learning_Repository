"""The small, project-specific NER output contract.

Qwen3 non-thinking chat format: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
"""

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
LABELS = {"PERSON", "ORGANIZATION", "LOCATION"}
SYSTEM_PROMPT = (
    "Extract all person, organization, and location names from the sentence. "
    "Copy each name exactly. Return only JSON in this format: "
    '{"entities": [{"text": "name", "label": "PERSON"}]}. '
    "Use only PERSON, ORGANIZATION, or LOCATION. "
    'If there are no entities, return {"entities": []}.'
)
