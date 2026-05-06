# Research Agent Smoke Test Prompts

Purpose:
Manual smoke-testing scenarios for `query_handler_node` + `research_agent_node`.

These prompts validate:
- deterministic routing
- clarification handling
- research execution
- degraded fallback behavior
- source population
- cost tracking
- graph stability

---

# 1. Standard Research Request

## Prompt

```text
research AI content marketing trends for 2026
```

## Expected Result

```text
intent = research
requested_outputs = ['research']
research_required = True
routing_decision = research_agent_node
workflow_status = research_complete

research_data exists = True
quality = degraded OR standard
source_count > 0
errors = []
```

---

# 2. Multi-Topic Research

## Prompt

```text
research the impact of AI agents on ecommerce, SEO, and social media marketing
```

## Expected Result

```text
intent = research
requested_outputs = ['research']
research_required = True

research_data populated
keywords populated
key_facts populated
source_count > 0
workflow_status = research_complete
```

---

# 3. Ambiguous Research Query

## Prompt

```text
AI trends
```

## Expected Result

```text
intent = clarification
clarification_needed = True
routing_decision = clarification_node

research_required = False
search_queries_used_this_session = 0
errors = []
```

---

# 4. Research + LinkedIn Content

## Prompt

```text
write a linkedin post about AI content marketing trends
```

## Expected Result

```text
intent = content_creation
requested_outputs includes linkedin
research_required = True
routing_decision = research_agent_node

research_data populated
source_count > 0
workflow_status = research_complete
```

---

# 5. Research + Blog Content

## Prompt

```text
create a blog article about future AI workflows in marketing agencies
```

## Expected Result

```text
requested_outputs includes blog
research_required = True
routing_decision = research_agent_node

research_data populated
errors = []
```

---

# 6. Research + Image Combination

## Prompt

```text
research futuristic fashion design trends and generate image concepts
```

## Expected Result

```text
requested_outputs includes image
research_required = True

routing remains deterministic
workflow_status != failed
errors = []
```

---

# 7. Garbage / Invalid Input

## Prompt

```text
asdfasdfasdf
```

## Expected Result

```text
clarification_needed = True
OR degraded research fallback

system does not crash
errors remain non-fatal
```

---

# 8. Short Clarification Query

## Prompt

```text
marketing
```

## Expected Result

```text
clarification_needed = True
routing_decision = clarification_node

research_required = False
```

---

# 9. Degraded Research Scenario

## Prompt

```text
research obscure synthetic biology fashion fabrics
```

## Expected Result

```text
quality = degraded

research_data still populated
summary non-empty
keywords non-empty
source snippets non-empty

cache should remain empty for degraded research
```

---

# 10. Query Expansion / Cost Control

## Prompt

```text
research AI trends in:
- fintech
- healthcare
- ecommerce
- logistics
- education
```

## Expected Result

```text
multiple search queries attempted

search_queries_used_this_session increases
workflow completes successfully
errors = []
```

---

# Smoke Test Success Criteria

For all successful runs:

```text
✔ no crashes
✔ no InvalidUpdateError
✔ deterministic routing
✔ workflow_status populated
✔ research_data populated when applicable
✔ source snippets populated
✔ errors remain empty or non-fatal
✔ cost controls update correctly
```

---

# Known Temporary Limitations

Current implementation may still exhibit:

- degraded fallback summaries
- verbose fallback key_facts
- stopwords in keywords
- placeholder final_response values
- LinkedIn-only requests temporarily including blog outputs

These are acceptable during current development phase.