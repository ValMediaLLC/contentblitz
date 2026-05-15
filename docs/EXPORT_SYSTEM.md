# Export System

## Scope

This document describes the implemented ContentBlitz export pipeline and metadata behavior.

Primary implementation:

- `contentblitz/agents/export_node.py`
- `contentblitz/tools/exports/markdown.py`
- `contentblitz/tools/exports/html.py`
- `contentblitz/tools/exports/pdf.py`
- `contentblitz/tools/exports/docx.py`
- `contentblitz/tools/exports/validation.py`
- `contentblitz/tools/exports/filenames.py`

## Supported Formats

- `markdown`
- `html`
- `pdf`
- `docx`

## Export Triggering

- Exports run from `export_node` during workflow execution.
- Export behavior is gated by `export_requested` and `export_metadata.formats_requested`.
- If `export_requested=true` and no formats are set, markdown is used as default.
- Restore does not regenerate exports.

## Pipeline Flow

1. Build assembled response in `output_assembler_node`.
2. Build format payloads in `export_node` using export renderers.
3. Sanitize payloads before delivery.
4. Run format-specific validation.
5. Persist only safe status/path metadata for successful formats.

## Validation Rules (Implemented)

Validation checks include:

- required document structure per format
- unsafe links/schemes
- script/unsafe embed markers
- stack traces and raw provider payload markers
- API key / env-var marker leakage
- base64/data-image leakage
- missing source section when sources are expected
- malformed/too-small binary payload checks for PDF/DOCX

## Export Metadata Shape

`export_metadata` contains:

- `formats_requested`
- `export_paths`
- `export_status`
- `error_log`
- `status_messages`
- `export_error_count`
- `exported_at`

Successful format example:

```json
{
  "export_status": { "markdown": "completed" },
  "export_paths": { "markdown": "exports/content_abcd1234.md" }
}
```

Failed format example:

```json
{
  "export_status": { "markdown": "failed" }
}
```

## Failure Semantics

- Validation/export failures are format-specific.
- One failed format does not automatically fail other requested formats.
- Workflow output remains available even when export formats fail.
- Export failures are surfaced as safe non-blocking warnings where applicable.

## Path and Filename Safety

- File names are deterministic hash-based names per format.
- Extensions are fixed by format (`.md`, `.html`, `.pdf`, `.docx`).
- Export paths are constrained to configured export directory.
- Path traversal outside export directory is rejected.

## Security/Sanitization Guarantees

Exports do not include:

- raw stack traces
- API keys or env var values/names
- raw provider config payloads
- base64 image blobs (`data:image/...`)

Image output failures are normalized to safe user-facing messages.
