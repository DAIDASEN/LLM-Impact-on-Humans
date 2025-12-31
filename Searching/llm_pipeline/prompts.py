from __future__ import annotations


VALIDATOR_SYSTEM = "You are a strict paper validator."

def validator_user_prompt(*, pdf_text: str, search_record_json: str) -> str:
    return f"""# Role: Strict Paper Validator

# Task:
1. Verify if the provided `PDF_TEXT` is a legitimate academic paper (look for signals like Abstract, Sections, References, etc.).
2. Verify if the **Title** and **Authors** in the `PDF_TEXT` match the provided `SEARCH_RECORD`.

# Instructions:
- Thinking Process: Provide a brief step-by-step analysis comparing the document structure and identity information.
- Final Output: You MUST end your response with a JSON block containing ONLY the following two fields:

```json
{{
  "is_paper": boolean,
  "is_match": boolean
}}
```

SEARCH_RECORD:
{search_record_json}

PDF_TEXT:
{pdf_text}
"""


SCREEN_SYSTEM = "You are a strict empirical-screening and clustering assistant."

def screen_user_prompt(*, pdf_text: str, search_record_json: str) -> str:
    return f"""# Role: Strict Empirical-Screening and Clustering Assistant

# Task:
Analyze the provided `PDF_TEXT` against the `SEARCH_RECORD` to determine if it is an empirical study on LLM impacts on humans and assign it to the correct cluster.

# Empirical Inclusion & Exclusion Rules:
1. MUST BE EMPIRICAL: Must contain human-subject experiments, field studies, RCTs, or analysis of human-generated content.
2. STRICT EXCLUSION:
   - Exclude purely theoretical, conceptual, or qualitative commentary papers without empirical data.
   - Exclude technical model evaluations (benchmarks/accuracy) with no human outcome.
   - Exclude system/tool papers with no user evaluation data.
3. SCOPE: Must study LLM impact on human behavior, decisions, attitudes, learning, productivity, etc.

# Clusters:
- 0: "Social & Collaboration" (Relationship, companionship, teamwork)
- 1: "Psychology & Persuasion" (Attitude change, trust, misinformation, bias)
- 2: "Creativity & Ideation" (Brainstorming, novelty, co-creation)
- 3: "Education & Productivity" (Learning, task performance, writing/coding assistance)
- 4: "None of the above" (Fits empirical scope but not these specific clusters)
- -1: "Excluded" (Not empirical, purely theoretical, or out of scope)

# Instructions:
- Thinking Process: You may provide a brief analysis of the paper's methodology and focus before the JSON block.
- Final Output: You MUST provide a JSON block at the end with this exact structure:

```json
{{
  "cluster_id": -1 | 0 | 1 | 2 | 3 | 4,
  "custom_summary": "If cluster_id is -1, explain why. If 4, summarize the research direction. If 0-3, provide a brief summary."
}}
```

SEARCH_RECORD:
{search_record_json}

PDF_TEXT:
{pdf_text}
"""


EXTRACT_SYSTEM = "You are an empirical information extractor focusing on LLM-human impact."

def extract_user_prompt(*, pdf_text: str, cluster_assignment_json: str) -> str:
    return f"""# Role: Empirical Information Extractor (LLM-Human Impact)

# Task:
Extract specific metadata and findings from the `PDF_TEXT` based on the previously determined `CLUSTER_ASSIGNMENT_JSON`. Focus exclusively on how LLMs impact humans.

# Extraction Rules:
1. Human-Centric Only: Ignore purely technical model metrics (e.g., perplexity, benchmarks). Only extract findings related to human behavior, attitudes, learning, productivity, etc.
2. Data Integrity: Use "NA" or null if the information is not explicitly supported by the text. Do not hallucinate.
3. Specifics:
   - LLM: Identify specific models (e.g., GPT-4, Llama-3).
   - Human_N: Capture the sample size and unit (participants, users, posts).
   - Long-term: Look for longitudinal data or follow-ups.

# Output Format:
You MUST provide a brief thinking process and then a single JSON block at the end:

```json
{{
  "Title": "Full paper title",
  "Cluster": [0,1,2,3] | [],
  "Keywords": ["8-15 specific phrases capturing domain + outcome + method"],
  "LLM": ["Model names"] | "NA",
  "Human_N": {{
    "value": integer | null,
    "unit": "participants" | "posts" | "users" | "documents" | "other" | "NA",
    "notes": "Brief context about the sample"
  }},
  "Interaction": "1-3 sentences describing how humans and LLM interact",
  "Long_term": "Duration of follow-up or NA",
  "Conclusion": ["2-6 bullet points on LLM-to-human impact"]
}}
```

CLUSTER_ASSIGNMENT_JSON:
{cluster_assignment_json}

PDF_TEXT:
{pdf_text}
"""



