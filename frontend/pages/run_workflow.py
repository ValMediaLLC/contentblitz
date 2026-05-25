"""Page for running ContentBlitz orchestration."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
from time import monotonic, sleep
from typing import Any

import streamlit as st

from contentblitz.ui.rendering import build_render_payload
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_status_messages,
    derive_node_statuses,
    summarize_workflow_status,
    workflow_requires_clarification,
)
from frontend.components.result_view import (
    render_collapsible_output_sections,
    render_degraded_and_error_state,
    render_execution_indicators,
    render_node_execution_statuses,
)
from frontend.config import FRONTEND_CONFIG
from frontend.services.orchestrator_client import stream_workflow_progress
from frontend.session import (
    add_history_entry,
    clear_persistence_messages,
    get_execution_status,
    get_last_error,
    get_last_result,
    get_last_submission,
    get_node_statuses,
    get_persistence_messages,
    get_progress_events,
    get_status_messages,
    save_persisted_run,
    set_execution_status,
    set_last_error,
    set_last_result,
    set_last_submission,
    set_node_statuses,
    set_progress_events,
    set_status_messages,
)

_WAITING_FOR_EVENT_MESSAGE = "Waiting for first workflow event..."
_QUERY_INPUT_KEY = "cbx_query_input"
_QUICK_START_KEY = "cbx_quick_start"
_QUICK_START_PROMPTS: dict[str, str] = {
    "Blog Article": (
        "Write a research-backed blog article about AI workflow automation for "
        "marketing teams. Include key trends, practical examples, and a clear "
        "implementation checklist."
    ),
    "LinkedIn Post": (
        "Write a concise LinkedIn post about how AI content workflows reduce "
        "rework for marketing teams, with a strong hook and 4-5 relevant hashtags."
    ),
    "Research Report": (
        "Produce a research report on 2026 AI content operations trends for B2B "
        "companies, including market signals, risks, and recommended next steps."
    ),
    "Image Concepts": (
        "Generate image concepts for a futuristic marketing campaign featuring "
        "AI-powered content operations dashboards, cinematic lighting, and clean "
        "brand styling."
    ),
    "Multi-Format Campaign": (
        "Create a multi-format campaign package about AI-native marketing "
        "workflows: a blog article, LinkedIn post, research summary, and image "
        "concepts, plus export-ready Markdown and PDF outputs."
    ),
}
_EMPTY_STATE_IDEAS: tuple[tuple[str, str], ...] = (
    (
        "Thought Leadership Launch",
        "Blog + LinkedIn + export pack for an AI workflow operations point of view.",
    ),
    (
        "Weekly Research Brief",
        "Concise trend summary for marketing leaders with citations and action steps.",
    ),
    (
        "Campaign Concept Sprint",
        "Multi-format narrative with image concepts for a single campaign theme.",
    ),
)


def _is_workflow_started_message(message: str) -> bool:
    normalized = str(message).strip().lower()
    if not normalized:
        return False
    return normalized.rstrip(".:!") == "workflow started" or normalized.startswith(
        "workflow started "
    )


def _execution_status_from_result(
    *,
    result: dict[str, object],
    node_statuses: dict[str, str],
) -> str:
    normalized_statuses = apply_optional_node_skips(
        state=result,
        node_statuses=node_statuses,
    )
    clarification_required = workflow_requires_clarification(
        state=result,
        node_statuses=normalized_statuses,
    )
    return summarize_workflow_status(
        normalized_statuses,
        workflow_status=str(result.get("workflow_status", "")).strip(),
        clarification_required=clarification_required,
    )


def _validate_submission_inputs(
    *,
    safe_query: str,
) -> str:
    if not safe_query:
        return "A prompt is required before running the workflow."
    return ""


def _render_active_execution_state(
    *,
    container: Any | None,
    execution_status: str,
    progress_events: list[dict[str, Any]],
    status_messages: list[str],
    node_statuses: dict[str, str] | None = None,
) -> None:
    if container is not None:
        container.empty()
        host = container.container()
    else:
        host = st.container()
    with host:
        with st.expander("Workflow", expanded=True):
            render_execution_indicators(
                execution_status=execution_status,
                result={"ui_workflow_status": execution_status, "routing_decision": ""},
                progress_events=progress_events,
            )
            render_node_execution_statuses(
                progress_events,
                live_timers=True,
                empty_message=_WAITING_FOR_EVENT_MESSAGE,
                node_statuses=node_statuses or {},
                show_full_orchestration_graph=True,
            )
            rendered_messages: set[str] = set()
            for message in status_messages:
                safe_message = str(message).strip()
                if not safe_message or safe_message.lower() in {"none", "null"}:
                    continue
                if _is_workflow_started_message(safe_message):
                    continue
                if safe_message in rendered_messages:
                    continue
                rendered_messages.add(safe_message)
                st.info(safe_message)


def _extract_submission_options_from_result(
    result: dict[str, object],
) -> dict[str, Any]:
    requested_outputs = [
        str(item).strip()
        for item in result.get("requested_outputs", [])
        if str(item).strip()
    ]
    export_requested = bool(result.get("export_requested", False))
    export_metadata = (
        result.get("export_metadata", {})
        if isinstance(result.get("export_metadata", {}), dict)
        else {}
    )
    formats_requested = [
        str(item).strip().lower()
        for item in export_metadata.get("formats_requested", [])
        if str(item).strip()
    ]
    if not export_requested:
        formats_requested = []
    return {
        "requested_outputs": requested_outputs,
        "export_requested": export_requested,
        "export_formats": formats_requested,
    }


def _set_quick_start_prompt_from_selection() -> None:
    selected = st.session_state.get(_QUICK_START_KEY)
    if not isinstance(selected, str):
        return
    prompt = _QUICK_START_PROMPTS.get(selected, "").strip()
    if prompt:
        st.session_state[_QUERY_INPUT_KEY] = prompt


def _render_idle_examples() -> None:
    cards = "".join(
        (
            "<article class='cbx-idle-example-card'>"
            f"<p class='cbx-idle-example-title'>{title}</p>"
            f"<p class='cbx-idle-example-copy'>{summary}</p>"
            "</article>"
        )
        for title, summary in _EMPTY_STATE_IDEAS
    )
    st.markdown(
        (
            "<section class='cbx-idle-examples'>"
            "<p class='cbx-idle-examples-title'>Recent workflow ideas</p>"
            "<div class='cbx-idle-examples-grid'>"
            f"{cards}"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render() -> None:
    if _QUERY_INPUT_KEY not in st.session_state:
        st.session_state[_QUERY_INPUT_KEY] = ""

    with st.container(border=True, key="cbx_run_form_card"):
        st.markdown(
            (
                "<p class='cbx-form-header'>Describe your content request</p>"
                "<p class='cbx-form-helper'>"
                "Generate blogs, LinkedIn posts, research reports, image concepts, "
                "and export-ready deliverables."
                "</p>"
            ),
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "Describe your content request",
            placeholder=FRONTEND_CONFIG.default_query_placeholder,
            height=116,
            key=_QUERY_INPUT_KEY,
            label_visibility="collapsed",
        )
        st.pills(
            "Quick-start suggestions",
            options=list(_QUICK_START_PROMPTS.keys()),
            selection_mode="single",
            default=None,
            key=_QUICK_START_KEY,
            on_change=_set_quick_start_prompt_from_selection,
            label_visibility="collapsed",
            width="stretch",
        )
        run_clicked = st.button(
            "Generate Content",
            type="primary",
            key="cbx_run_submit",
        )

    if run_clicked:
        safe_query = str(query).strip()
        set_last_submission({"user_query": safe_query})

        validation_error = _validate_submission_inputs(
            safe_query=safe_query,
        )
        if validation_error:
            set_execution_status("idle")
            set_last_error(validation_error)
        else:
            live_execution_placeholder = st.empty()
            progress_events: list[dict[str, Any]] = []
            try:
                set_execution_status("running")
                set_progress_events(progress_events)
                node_statuses = derive_node_statuses(progress_events)
                set_node_statuses(node_statuses)
                set_status_messages([])
                _render_active_execution_state(
                    container=live_execution_placeholder,
                    execution_status="running",
                    progress_events=progress_events,
                    status_messages=get_status_messages(),
                    node_statuses=node_statuses,
                )
                final_result: dict[str, object] = {}
                event_queue: Queue[dict[str, Any]] = Queue()

                def _consume_stream() -> None:
                    try:
                        for stream_event in stream_workflow_progress(
                            user_query=safe_query,
                        ):
                            if isinstance(stream_event, dict):
                                event_queue.put(dict(stream_event))
                    except Exception as stream_error:  # pragma: no cover - safety
                        event_queue.put(
                            {"type": "error", "error_message": str(stream_error)}
                        )
                    finally:
                        event_queue.put({"type": "done"})

                stream_thread = Thread(target=_consume_stream, daemon=True)
                stream_thread.start()

                last_live_render_at = monotonic()
                stream_finished = False
                while not stream_finished:
                    consumed_event = False
                    while True:
                        try:
                            event = event_queue.get_nowait()
                        except Empty:
                            break
                        consumed_event = True
                        event_type = str(event.get("type", "")).strip().lower()
                        if event_type == "progress":
                            raw_event = event.get("event")
                            if isinstance(raw_event, dict):
                                progress_events.append(dict(raw_event))
                                set_progress_events(progress_events)
                                node_statuses = derive_node_statuses(progress_events)
                                set_node_statuses(node_statuses)
                        elif event_type == "final":
                            result_payload = event.get("result")
                            if isinstance(result_payload, dict):
                                final_result = result_payload
                                raw_events = event.get("events")
                                if isinstance(raw_events, list):
                                    progress_events = [
                                        dict(item)
                                        for item in raw_events
                                        if isinstance(item, dict)
                                    ]
                                    set_progress_events(progress_events)
                                    node_statuses = derive_node_statuses(
                                        progress_events
                                    )
                                    set_node_statuses(node_statuses)
                        elif event_type == "error":
                            message = str(event.get("error_message", "")).strip()
                            raise RuntimeError(message or "Unknown stream error")
                        elif event_type == "done":
                            stream_finished = True

                    current_time = monotonic()
                    render_tick_elapsed = current_time - last_live_render_at
                    should_live_tick = (
                        bool(progress_events)
                        and get_execution_status() == "running"
                        and render_tick_elapsed >= 0.1
                    )
                    if consumed_event or should_live_tick:
                        _render_active_execution_state(
                            container=live_execution_placeholder,
                            execution_status="running",
                            progress_events=progress_events,
                            status_messages=get_status_messages(),
                            node_statuses=node_statuses,
                        )
                        last_live_render_at = current_time

                    if stream_finished:
                        break
                    sleep(0.1)

                if stream_thread.is_alive():
                    stream_thread.join(timeout=0.5)

                live_execution_placeholder.empty()
                normalized_node_statuses = apply_optional_node_skips(
                    state=final_result,
                    node_statuses=node_statuses,
                )
                set_node_statuses(normalized_node_statuses)
                ui_workflow_status = _execution_status_from_result(
                    result=final_result,
                    node_statuses=normalized_node_statuses,
                )
                final_result["ui_workflow_status"] = ui_workflow_status
                final_result["ui_node_statuses"] = normalized_node_statuses
                selected_options = _extract_submission_options_from_result(final_result)
                final_result["ui_selected_options"] = {
                    **selected_options,
                    "user_query": safe_query,
                }
                computed_status_messages = build_status_messages(
                    state=final_result,
                    node_statuses=normalized_node_statuses,
                )
                final_result["status_messages"] = computed_status_messages
                final_result["ui_progress_events"] = progress_events

                set_last_result(final_result)
                set_execution_status(ui_workflow_status)
                set_status_messages(computed_status_messages)
                set_last_submission(final_result["ui_selected_options"])
                persisted_run_id = save_persisted_run(
                    result=final_result,
                    last_submission=final_result["ui_selected_options"],
                    progress_events=progress_events,
                    node_statuses=normalized_node_statuses,
                    status_messages=computed_status_messages,
                )
                if persisted_run_id:
                    final_result["run_id"] = persisted_run_id
                    set_last_result(final_result)
                add_history_entry(
                    user_query=safe_query,
                    requested_outputs=selected_options["requested_outputs"],
                    workflow_status=str(
                        final_result.get("ui_workflow_status", "")
                        or final_result.get("workflow_status", "")
                    ).strip(),
                )
            except Exception:
                set_execution_status("failed")
                set_last_error("Workflow execution failed. Check logs for details.")
                set_status_messages(["Workflow execution failed before completion."])
                live_execution_placeholder.empty()

    last_error = get_last_error()
    if last_error:
        st.error(last_error)
    persistence_messages = get_persistence_messages()
    for message in persistence_messages:
        st.warning(message)
    if persistence_messages:
        clear_persistence_messages()

    result = get_last_result()
    execution_status = get_execution_status()
    result_for_indicators = result or {}
    if isinstance(result, dict):
        normalized_indicator_statuses = apply_optional_node_skips(
            state=result,
            node_statuses=get_node_statuses(),
        )
        clarification_required = workflow_requires_clarification(
            state=result,
            node_statuses=normalized_indicator_statuses,
        )
        indicator_workflow_status = summarize_workflow_status(
            normalized_indicator_statuses,
            workflow_status=str(result.get("workflow_status", "")).strip(),
            clarification_required=clarification_required,
        )
        result_for_indicators = dict(result)
        result_for_indicators["ui_workflow_status"] = indicator_workflow_status
    progress_events = get_progress_events()
    node_statuses = get_node_statuses()
    if not node_statuses and progress_events:
        node_statuses = derive_node_statuses(progress_events)
    if isinstance(result, dict):
        node_statuses = apply_optional_node_skips(
            state=result, node_statuses=node_statuses
        )

    result = get_last_result()
    if not result:
        has_submitted_run = bool(
            str(get_last_submission().get("user_query", "")).strip()
        )
        show_active_state = execution_status in {
            "running",
            "failed",
            "partial_success",
            "success",
            "awaiting_clarification",
        }
        if show_active_state and has_submitted_run:
            _render_active_execution_state(
                container=None,
                execution_status=execution_status,
                progress_events=progress_events,
                status_messages=get_status_messages(),
                node_statuses=node_statuses,
            )
            return
        _render_idle_examples()
        return

    render_payload = build_render_payload(
        state=result,
        node_statuses=node_statuses,
    )
    render_degraded_and_error_state(render_payload)
    render_collapsible_output_sections(
        render_payload=render_payload,
        status_messages=get_status_messages(),
        execution_status=execution_status,
        indicator_result=result_for_indicators
        if isinstance(result_for_indicators, dict)
        else {},
        node_statuses=node_statuses,
        progress_events=progress_events,
        raw_state=result,
        raw_submission=get_last_submission(),
    )
