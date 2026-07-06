# Attribute Knowledge RAG (AK-RAG)

Attribute Knowledge RAG (AK-RAG) is a reference architecture for transforming enterprise attribute metadata into an AI-searchable knowledge layer.

Instead of embedding PDFs, wiki pages, or patient rows, AK-RAG indexes clinical and operational attributes as individual knowledge objects. Each attribute document contains its business name, technical field, synonyms, allowed values, ranges, rules, governance metadata, and embedding text. This lets a system translate natural language into approved clinical or operational attributes with lower hallucination risk and clearer user clarification paths.

This repository demonstrates the pattern on a healthcare cohort-definition use case, but the underlying idea is domain-agnostic: **any enterprise that needs an agent to translate natural language into a governed set of attributes — deterministically, with no freeform field invention — can apply this pattern.** Banks, insurers, retailers, and other regulated or metadata-heavy industries can swap in their own attribute catalog (risk factors, KYC fields, product attributes, policy terms) and get the same guarantee: every output is a real, governed `attribute_id`, never a hallucinated field. See [Beyond Healthcare: A General Pattern for Governed Agentic Workflows](#beyond-healthcare-a-general-pattern-for-governed-agentic-workflows) below.

## Demo

`data/demo.pdf` walks through a live run of the reference UI end to end:

1. **Natural language** — the user asks for a population, e.g. `diabetic patients over 65 with elevated HbA1c`.
2. **LLM phrase extraction** — the LLM splits the request into individual attribute phrases and nothing else.
3. **Hybrid retrieval** — BM25 + kNN vector search run per phrase, fused with Reciprocal Rank Fusion.
4. **Decision policy** — each phrase is classified exact / near / ambiguous / none against fixed thresholds.
5. **Clarify or resolve** — non-exact phrases ask the user to choose; the system never guesses.
6. **Governed DSL** — only `attribute_id`s that exist in the catalog can appear in the final filter, e.g.:

   ```json
   {
     "version": "1.0",
     "type": "attribute_filter",
     "filters": [
       {
         "attribute_id": "clinical.risk_score",
         "business_name": "Clinical Risk Score",
         "technical_field": "clinical_risk_score_num",
         "source_phrase": "12 months"
       }
     ]
   }
   ```

The healthcare example is one instantiation of this pattern — the same six-step pipeline (parse → retrieve → classify → clarify → govern → emit DSL) applies unchanged if the catalog is swapped for a different industry's governed attributes.

## Problem

Healthcare analysts and care managers often describe populations in language that does not exactly match enterprise metadata.

```text
User:
Diabetic patients with high HbA1c who were readmitted in the last 30 days
```

The enterprise system may not have a `high HbA1c` attribute. It may have `HbA1c > 8.0` and `HbA1c > 9.0` thresholds governed separately. It may also need to disambiguate whether `readmitted in the last 30 days` refers to all-cause readmission or condition-specific readmission.

AK-RAG treats this as an attribute selection problem, not a free-form generation problem.

```text
System:
"high HbA1c" may map to one of the following governed thresholds:
1. HbA1c Level > 8.0% (poor control)
2. HbA1c Level > 9.0% (very poor control)

"readmitted in the last 30 days" maps to 30-Day All-Cause Readmission.

Which HbA1c threshold should be used?
```

After the user chooses:

```text
User:
HbA1c over 9

System:
HbA1c Level > 9.0% AND 30-Day All-Cause Readmission = true.
Estimated population: 4,200 members.
Save and refine?
```

## Core Idea

- Excel is the authoring input.
- NDJSON is the ingestion format.
- OpenSearch is the serving layer.
- The LLM interprets the user request and manages clarification.
- Retrieval happens one attribute phrase at a time.
- One embedding is created per attribute document, not per patient row.
- Hybrid search always retrieves candidates using BM25 + kNN with Reciprocal Rank Fusion (RRF).
- The decision engine classifies each phrase as exact, near, ambiguous, or none.
- The system guides the conversation instead of inventing unsupported fields.

## Architecture

AK-RAG has three pluggable provider layers — LLM, embedding, and search backend. All three are selected at startup via environment variables and can be swapped without code changes.

### Ingestion Pipeline

```text
Excel / CSV / API
        |
        v
Contract Validation & Normalization
        |
        v
NDJSON  (One Attribute = One Document)
        |
        v
+--------------------------------------------------+
|  Embedding Provider    EMBEDDING_PROVIDER=...    |
|                                                  |
|  sentence-transformers (dev default, no API key) |
|  openai · gemini · bedrock  (production)         |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|  Search / Index Backend    SEARCH_PROVIDER=...   |
|                                                  |
|  local      BM25-style + token vector + RRF      |
|             offline, no cluster  (dev default)   |
|  opensearch BM25 + HNSW kNN + RRF  (production) |
|  faiss      dense vector, no cluster             |
|  chroma     dense vector + metadata filtering    |
+--------------------------------------------------+
```

### Query / Conversation Pipeline

```text
User Natural Language Input
        |
        v
+--------------------------------------------------+
|  LLM Provider    LLM_PROVIDER=...               |
|                                                  |
|  claude (default) · openai · bedrock · ollama   |
|  → extract attribute phrases from the request   |
+--------------------------------------------------+
        |
        v
For each phrase:
  Embedding Provider  →  encode phrase
        |
  Search / Index Backend  →  hybrid retrieval
        |
  Exact / Near / Ambiguous / None
  Decision Engine
        |
+--------------------------------------------------+
|  LLM Provider  →  generate clarification dialog |
+--------------------------------------------------+
        |
        v
Governance Check
PHI  ·  HIPAA category  ·  consent  ·  minimum cell size
        |
        v
DSL / Query / Cohort Definition
        |
        v
Enterprise / Clinical Applications
```

## Pluggable Providers

### Provider Matrix

| Layer | Provider | Notes |
| --- | --- | --- |
| LLM | `claude` | Default. Phrase extraction and clarification dialog. Requires `ANTHROPIC_API_KEY`. |
| LLM | `openai` | Alternative hosted LLM. Requires `OPENAI_API_KEY`. |
| LLM | `bedrock` | AWS-hosted. No API key rotation. Requires AWS credentials. |
| LLM | `ollama` | Local or air-gapped. No API cost. Requires a local Ollama server. |
| Embedding | `sentence` | Dev default. No API key. Uses `all-MiniLM-L6-v2`. Fast and small. |
| Embedding | `openai` | Production quality dense vectors. Requires `OPENAI_API_KEY`. |
| Embedding | `gemini` | Production alternative. Requires `GOOGLE_API_KEY`. |
| Embedding | `bedrock` | AWS-hosted embedding. Requires AWS credentials. |
| Search | `local` | Dev default. Offline. BM25-style lexical + token vector + RRF. No cluster needed. |
| Search | `opensearch` | Production. Full BM25 + HNSW kNN + RRF. Requires OpenSearch cluster. |
| Search | `faiss` | Dense vector only. No cluster. Suitable for moderate-scale datasets. |
| Search | `chroma` | Dense vector with metadata filtering. Embedded or hosted. |

### Recommended Combinations

| Environment | LLM | Embedding | Search |
| --- | --- | --- | --- |
| Local dev | `claude` or `ollama` | `sentence` | `local` |
| CI / testing | `claude` | `sentence` | `local` |
| Production (AWS) | `bedrock` | `bedrock` | `opensearch` |
| Production (general) | `claude` | `openai` | `opensearch` |
| Air-gapped / on-prem | `ollama` | `sentence` | `faiss` or `opensearch` |

### Configuration

All providers are set in `.env`:

```env
# LLM provider: claude | openai | bedrock | ollama
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...

# Embedding provider: sentence | openai | gemini | bedrock
EMBEDDING_PROVIDER=sentence

# Search backend: local | opensearch | faiss | chroma
SEARCH_PROVIDER=local

# Decision thresholds
EXACT_THRESHOLD=0.92
NEAR_THRESHOLD=0.75
RRF_K=60

# OpenSearch connection (when SEARCH_PROVIDER=opensearch)
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
```

### Provider Installs

```bash
pip install -e .                # base: local search, no LLM, no embedding API
pip install -e ".[claude]"      # + Anthropic SDK  (claude LLM)
pip install -e ".[openai]"      # + OpenAI SDK     (openai LLM + openai embedding)
pip install -e ".[bedrock]"     # + boto3          (bedrock LLM + embedding)
pip install -e ".[sentence]"    # + sentence-transformers
pip install -e ".[gemini]"      # + google-generativeai
pip install -e ".[opensearch]"  # + opensearch-py
pip install -e ".[faiss]"       # + faiss-cpu
pip install -e ".[chroma]"      # + chromadb
pip install -e ".[excel]"       # + openpyxl (XLSX file support)
pip install -e ".[all]"         # everything
```

## Ingestion Pipeline

1. Read Excel.
2. Validate every row against the contract schema.
3. Normalize fields:
   - split synonyms
   - parse ranges
   - coerce data types
   - standardize units, dates, windows, and categories
4. Compose embedding text, for example:

   ```text
   type | business_name | technical_field | synonyms | definition | rules | values | domain
   ```

5. Generate embeddings in batches.
6. Create a new versioned OpenSearch index:

   ```text
   attributes_vYYYY_MM_DD
   ```

7. Bulk index documents using `attribute_id` as `_id` for idempotency.
8. Run smoke tests with fixed canonical queries.
9. Atomically swap the `attributes_current` alias to the new index.
10. Roll back by pointing the alias back to the previous index if validation fails.

## Why NDJSON?

NDJSON is useful for this pipeline because each attribute is represented as one independent JSON document per line.

- Works naturally with OpenSearch bulk indexing.
- Streams well for large metadata files.
- Makes failed rows easier to isolate and retry.
- Avoids loading the full corpus into memory.
- Keeps ingestion idempotent when paired with stable `attribute_id` values.
- Supports versioned rebuilds without editing a live index in place.

## Search Algorithm

AK-RAG uses hybrid retrieval:

- BM25 for exact keyword, field, acronym, and synonym matches.
- kNN vector search for semantic similarity.
- RRF to combine ranked result lists.

RRF is preferred over raw score averaging because BM25 scores and vector similarity scores are not directly comparable. Fusion by rank is more stable across query types, embedding models, and index changes.

The system retrieves top-k candidates for each attribute phrase, then applies decision logic.

## Decision Policy

Each searched phrase is classified into one of four outcomes.

| Outcome | Behavior |
| --- | --- |
| Exact / high confidence | Use the attribute as-is. |
| Near match | Show top options and ask the user to choose. |
| Ambiguous | Ask a clarifying question before selecting. |
| None | Say no supported attribute was found and ask for an alternative or broader category. |

Examples in the healthcare domain:

- `high HbA1c` requires clarification between HbA1c > 8.0% and HbA1c > 9.0% thresholds.
- `recent ER visit` is ambiguous because `recent` has no fixed time window — the system must ask whether 30, 60, or 90 days is intended.
- `readmitted` may be a near match to `30-Day All-Cause Readmission` or `30-Day Condition-Specific Readmission`.
- `high-risk member` may map to risk stratification tier (e.g., tier 3 or 4) and must be confirmed.

## Hallucination Mitigation

AK-RAG reduces hallucination by constraining generation to retrieved, governed attributes.

- Search one attribute phrase at a time.
- Require exact, near, ambiguous, or none classification before using an attribute.
- Never invent unavailable clinical fields.
- Return supported alternatives when there is no exact match.
- Ask for clarification on vague quantifiers like `high`, `recent`, `active`, or `complex`.
- Ask for clarification when multiple governed attributes could match.
- Generate DSL only from selected attribute IDs, not from free-form model text.
- Preserve source metadata and confidence signals in every generated output.

## Healthcare Domain

### Clinical Attributes

Clinical attributes include lab values, vital signs, diagnoses, procedures, and risk scores. Examples:

| Attribute ID | Business Name | Notes |
| --- | --- | --- |
| `clinical.hba1c` | HbA1c Level | Multiple governed thresholds per clinical guideline |
| `clinical.bmi` | Body Mass Index | Continuous; obesity tiers require clarification |
| `clinical.sbp` | Systolic Blood Pressure | Hypertension staging uses distinct cutoffs |
| `diagnosis.diabetes_type2` | Type 2 Diabetes Diagnosis | ICD-10 coded; confirm active vs. historical |
| `procedure.annual_wellness_visit` | Annual Wellness Visit | Date-bounded; HEDIS measure alignment |

### Operational Attributes

Operational attributes cover utilization, enrollment, and administrative data. Examples:

| Attribute ID | Business Name | Notes |
| --- | --- | --- |
| `utilization.er_visits_90d` | ER Visits in Last 90 Days | Frequently asked; distinguish ED vs. observation |
| `utilization.readmission_30d` | 30-Day All-Cause Readmission | All-cause vs. condition-specific needs clarification |
| `enrollment.payer_type` | Payer / Insurance Type | Medicare, Medicaid, Commercial, Self-pay |
| `risk.chronic_condition_count` | Chronic Condition Count | Count of active chronic conditions on record |
| `risk.stratification_tier` | Risk Stratification Tier | Tiers 1–4 per health plan risk model |

### Healthcare Governance

Healthcare attributes carry additional governance requirements beyond standard data governance.

| Governance Field | Description |
| --- | --- |
| `phi` | Protected Health Information under HIPAA |
| `de_identification_required` | Whether output must be de-identified before sharing |
| `hipaa_category` | clinical, demographic, financial, operational |
| `minimum_cell_size` | Minimum cohort size before results can be returned |
| `allowed_channels` | care_management, quality_reporting, analytics, claims |
| `consent_required` | Whether member consent is required to use the attribute |

The governance layer checks these fields before DSL generation and blocks attributes that violate channel, consent, or de-identification rules.

### HEDIS and Quality Measures

AK-RAG can index HEDIS measure components as governed attributes. Each measure component is a first-class attribute with its own definition, allowed values, and governance metadata. The system can guide an analyst from natural language like `members who missed their annual diabetic eye exam` to the correct HEDIS denominator and numerator exclusion attributes without inventing unsupported fields.

### Vague Quantifiers in Clinical Language

Clinical natural language is especially prone to vague quantifiers that require governed clarification.

| Phrase | Clarification Needed |
| --- | --- |
| `high HbA1c` | > 8.0% or > 9.0%? |
| `recent ER visit` | 30, 60, or 90 days? |
| `high-risk member` | Risk tier 3+, 4 only, or a specific risk score threshold? |
| `complex patient` | Multiple chronic conditions? High utilization? Both? |
| `active diagnosis` | Active in last 12 months or currently on problem list? |
| `elderly` | Age ≥ 65 or ≥ 75? |

## OpenSearch Indexing Strategy

Indexes are immutable release artifacts.

```text
attributes_v2026_06_26
attributes_v2026_07_15
attributes_current -> attributes_v2026_07_15
```

The application reads from the `attributes_current` alias. New metadata releases are loaded into a fresh versioned index, tested, then promoted with an atomic alias swap.

This avoids partial updates, supports rollback, and makes embedding/model drift easier to manage.

## Attribute Document Shape

Example NDJSON record for a clinical attribute:

```json
{"attribute_id":"clinical.hba1c","type":"numeric","business_name":"HbA1c Level","technical_field":"hba1c_pct","domain":"clinical","definition":"Most recent HbA1c percentage on record. Indicates glycemic control. Used for diabetes population identification and quality reporting.","synonyms":["glycated hemoglobin","blood sugar control","a1c","hemoglobin a1c","glycohemoglobin"],"operators":[">",">=","<","<=","="],"unit":"%","example_values":[7.0,8.0,9.0,10.0],"governance":{"phi":true,"hipaa_category":"clinical","de_identification_required":false,"minimum_cell_size":11,"allowed_channels":["care_management","quality_reporting","analytics"],"consent_required":false},"embedding_model":"text-embedding-model-name","embedding_version":"2026-06-26","embedding_text":"numeric | HbA1c Level | hba1c_pct | glycated hemoglobin | blood sugar control | a1c | glycemic control | diabetes | percentage | clinical"}
```

Example NDJSON record for an operational attribute:

```json
{"attribute_id":"utilization.er_visits_90d","type":"numeric","business_name":"ER Visits in Last 90 Days","technical_field":"er_visit_count_90d","domain":"utilization","definition":"Count of emergency room visits in the last 90 days. Excludes observation stays.","synonyms":["emergency room visits","ED visits","emergency department visits","ER utilization","high ER use"],"operators":[">",">=","=","<","<="],"unit":"count","example_values":[1,2,3],"governance":{"phi":true,"hipaa_category":"operational","de_identification_required":false,"minimum_cell_size":11,"allowed_channels":["care_management","analytics","claims"],"consent_required":false},"embedding_model":"text-embedding-model-name","embedding_version":"2026-06-26","embedding_text":"numeric | ER Visits in Last 90 Days | er_visit_count_90d | emergency room | ED visits | high ER use | utilization | 90 days | count"}
```

## Main Risks And Decisions

| Risk | Decision |
| --- | --- |
| Excel as source of truth can be fragile | Enforce strict schema validation and consider a governed UI later. |
| Vague clinical quantifiers | Ask clarifying questions instead of guessing; never assume a threshold. |
| Ambiguous attributes | Force clarification, for example all-cause vs. condition-specific readmission. |
| PHI and HIPAA compliance | Tag every attribute with phi, hipaa_category, and minimum_cell_size; enforce before DSL generation. |
| Embedding model drift | Tag documents with model and embedding version; rebuild on upgrade. |
| Hybrid score pitfalls | Use RRF instead of raw score averaging. |
| Latency | Cache phrase embeddings and parallelize independent lookups. |
| Governance violations | Apply PHI, channel, consent, and de-identification rules before DSL generation. |
| Clinical guideline changes | Version indexes by date; rebuild when governed thresholds change. |

## Builder Order

1. Contract schema and Excel validator.
2. OpenSearch mapping, ingestion pipeline, versioned indexes, and alias swap.
3. Hybrid retriever using BM25 + kNN + RRF.
4. Exact / near / ambiguous / none decision engine.
5. Conversion orchestrator: phrase -> retrieval -> clarification -> DSL.
6. Governance layer for PHI, channel, consent, and minimum cell size rules.
7. Evaluation set, smoke tests, and metrics dashboard.

## Initial Implementation Scope

The first implementation should focus on a narrow but complete vertical slice:

1. Define the Excel contract.
2. Validate a sample workbook.
3. Produce NDJSON attribute documents.
4. Create a versioned OpenSearch index.
5. Index attributes idempotently.
6. Run hybrid retrieval for one phrase.
7. Classify retrieval result as exact, near, ambiguous, or none.
8. Produce a small DSL only after attributes are selected.

## Local Implementation

This repository includes a working Python vertical slice:

- Contract validation for CSV and `.xlsx` attribute files.
- NDJSON generation with one document per attribute.
- OpenSearch mapping generation for a versioned attribute index.
- Offline hybrid retrieval using BM25-style lexical ranking, token-vector similarity, and RRF.
- Exact / near / ambiguous / none decision policy.
- DSL assembly that only includes selected attributes and keeps unresolved phrases separate.

The local retriever is intentionally dependency-light so the behavior can be tested without a running OpenSearch cluster. Production search should use OpenSearch BM25 + HNSW kNN and the same decision policy.

## Repository Layout

```text
src/akrag/
  cli.py            Command-line entry point
  contract.py       Attribute contract validation
  io.py             CSV/XLSX/NDJSON readers and writers
  search.py         Local hybrid retriever
  decision.py       Exact / near / ambiguous / none policy
  orchestrator.py   Phrase evaluation and DSL assembly
  opensearch.py     Versioned index names and mapping helpers

data/
  sample_attributes.csv

config/
  opensearch_mapping.json

tests/
  test_contract.py
  test_retrieval.py
```

## Quick Start

Run tests:

```bash
python3 -m pytest
```

Validate the sample attribute contract:

```bash
PYTHONPATH=src python3 -m akrag.cli validate data/sample_attributes.csv
```

Convert the contract to NDJSON:

```bash
PYTHONPATH=src python3 -m akrag.cli to-ndjson data/sample_attributes.csv build/attributes.ndjson
```

Generate an OpenSearch mapping:

```bash
PYTHONPATH=src python3 -m akrag.cli mapping build/opensearch_mapping.json
```

Run local phrase retrieval and decisioning:

```bash
PYTHONPATH=src python3 -m akrag.cli query build/attributes.ndjson "high HbA1c" "readmitted in the last 30 days"
```

Example output:

```json
{
  "filters": [
    {
      "attribute_id": "utilization.readmission_30d",
      "business_name": "30-Day All-Cause Readmission",
      "source_phrase": "readmitted in the last 30 days"
    }
  ],
  "unresolved": [
    {
      "phrase": "high HbA1c",
      "outcome": "ambiguous",
      "options": [
        {
          "attribute_id": "clinical.hba1c_gt_8",
          "business_name": "HbA1c Level > 8.0%"
        },
        {
          "attribute_id": "clinical.hba1c_gt_9",
          "business_name": "HbA1c Level > 9.0%"
        }
      ]
    }
  ]
}
```

To install the CLI as `akrag`:

```bash
pip install -e ".[claude,sentence]"   # Claude LLM + sentence-transformers (recommended start)
akrag validate data/sample_attributes.csv
```

See the [Provider Installs](#provider-installs) section for the full list of provider-specific extras.

## Healthcare Use Cases

- **Population health management** — Identify cohorts for care management programs using clinical and utilization attributes.
- **Care gap identification** — Detect members missing HEDIS-aligned preventive care or chronic disease management.
- **Risk stratification** — Build high-risk cohorts using risk tier, chronic condition count, and utilization signals.
- **Quality measure reporting** — Compose HEDIS denominator and numerator definitions from governed attributes without free-form generation.
- **Prior authorization support** — Map clinical criteria in PA requests to governed diagnosis and procedure attributes.
- **Utilization management** — Identify high-utilization members using ER visit, admission, and readmission attributes.
- **Chronic disease management** — Define diabetic, cardiac, or respiratory cohorts from governed ICD-10 and lab value attributes.
- **Value-based care analytics** — Align attributed populations to cost and quality contracts using governed member attributes.

## Beyond Healthcare: A General Pattern for Governed Agentic Workflows

The healthcare walkthrough in `data/demo.pdf` is one instance of a general pattern: **natural language → LLM phrase extraction → hybrid retrieval → exact/near/ambiguous/none decision → governed DSL.** Nothing in the pipeline is healthcare-specific — `clinical.*` and `labs.*` are just attribute domains in the catalog. Any industry with a governed set of enterprise attributes and a need for agentic natural-language access can plug in its own catalog instead.

This matters most in regulated or high-stakes domains, where an agent that free-generates field names or thresholds is a compliance risk. AK-RAG constrains the agent to attributes that already exist and are already governed, so "the LLM invented a field that doesn't exist in our data model" stops being a failure mode.

### Banking Example

```text
User:
Retail customers with high credit utilization and a recent late payment

System:
"high credit utilization" may map to one of the following governed thresholds:
1. Credit Utilization Ratio > 30% (elevated)
2. Credit Utilization Ratio > 50% (high risk)

"recent late payment" has no fixed time window — did you mean the last 30, 60, or 90 days?
```

| Attribute ID | Business Name | Notes |
| --- | --- | --- |
| `risk.credit_utilization_ratio` | Credit Utilization Ratio | Continuous; risk tiers require clarification |
| `risk.late_payment_90d` | Late Payment in Last 90 Days | Window must be confirmed (30/60/90) |
| `kyc.customer_risk_tier` | KYC Customer Risk Tier | Tiers 1–4 per compliance risk model |
| `account.product_type` | Account / Product Type | Checking, savings, credit, loan |
| `aml.sar_flag_count` | SAR Flag Count | Count of active Suspicious Activity Report flags |

The same governance layer used for PHI in healthcare (`allowed_channels`, `consent_required`, `minimum_cell_size`) maps directly onto banking controls: PII/NPI masking, regulatory reporting eligibility, and minimum-cohort-size rules for fair lending analysis.

### Other Industries

- **Insurance** — policy terms, coverage tiers, claims attributes, underwriting risk factors.
- **Retail and loyalty** — customer segments, purchase behavior attributes, loyalty tier definitions.
- **Customer data platforms** — governed customer/account attributes for audience building without ad-hoc SQL.
- **Cybersecurity** — asset risk attributes, alert severity classifications, entitlement/access attributes.
- **Data catalogs** — general-purpose natural-language search over any governed metadata catalog.
- **HR systems** — compensation bands, performance attributes, org/role classifications.
- **Supply chain** — inventory, supplier risk, and logistics attributes with governed thresholds.

In every case the mechanics are identical to the healthcare demo: swap the attribute catalog, keep the pipeline (parse → retrieve → classify → clarify → govern → emit DSL) unchanged.

## Project Status

Early development.

## License

Apache License 2.0
