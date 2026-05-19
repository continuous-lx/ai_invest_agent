# Local RAG Knowledge Base

Place local text documents in `data/documents/`.

Supported file types:

- `.md`
- `.txt`

Company-specific documents must include a matching ticker symbol. They are only
retrieved when analyzing that same symbol.

Example company document:

```markdown
---
title: NVIDIA 2025 10-K Notes
symbol: NVDA
doc_type: 10-K
---

Revenue, margin, cash flow, risks, and management commentary notes...
```

You can also infer metadata from the filename:

```text
data/documents/NVDA_10K_2025.md
data/documents/AAPL_10Q_2026_Q1.txt
```

General documents can be used for any stock:

```markdown
---
title: Position Sizing Rules
scope: general
doc_type: investment_framework
---

Portfolio construction and risk management rules...
```

Or by filename:

```text
data/documents/general_investment_framework.md
```

Safety rule:

- `symbol: AAPL` documents are never used when analyzing `MSFT`.
- `scope: general` documents may be used for any ticker.
- If no matching company document exists, the researcher receives an explicit
  data gap note instead of using another company's filings.
