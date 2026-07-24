"""Central prompt assembly for the synchronous communication protocol."""

import json
from typing import Literal

from .state import CommunicationState


class PromptComposer:
    """Build role prompts while keeping context visibility explicit."""

    @staticmethod
    def _eligible_recipients(actor_id: str, state: CommunicationState) -> tuple[str, ...]:
        recipients = tuple(solver_id for solver_id in state.solver_ids if solver_id != actor_id)
        if not recipients:
            return recipients
        offset = state.actor_step_count(actor_id) % len(recipients)
        return recipients[offset:] + recipients[:offset]

    @staticmethod
    def _action_contract(
        *,
        role: str,
        eligible_recipient_ids: tuple[str, ...],
        final_answer_key: str,
        allow_ask: bool,
        requires_decision_provenance: bool = False,
    ) -> str:
        sections = [f"You are the {role}. Explore the problem and record a concise reasoning_note."]
        if allow_ask:
            sections.extend(
                [
                    "If you ask, ask one eligible solver about the assumption, derivation, counterexample, or fact you are least certain about.",
                    f"eligible_recipient_ids (ordered only for display): {list(eligible_recipient_ids)}. Choose by relevance, never by identifier order.",
                ]
            )
        sections.extend(
            [
                "Your entire response must be exactly one valid JSON object: start with `{` and end with `}`.",
                "Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys.",
            ]
        )
        forms = ['{"action":"think","reasoning_note":"brief private progress"}']
        if allow_ask:
            forms.append('{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}')
        if requires_decision_provenance:
            forms.extend(
                [
                    '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"exact answer","decision_mode":"select_candidate","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":"<solver_id>"}',
                    '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"corrected exact answer","decision_mode":"single_source_repair","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":"<solver_id>"}',
                    '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"exact answer","decision_mode":"rederive_from_trace","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":null}',
                ]
            )
        else:
            forms.append(f'{{"action":"submit","reasoning_note":"final concise rationale","{final_answer_key}":"exact answer"}}')
        sections.append(f"Choose exactly one valid form: {'; '.join(forms)}.")
        return " ".join(sections)

    @staticmethod
    def solver_step_prompt(*, actor_id: str, problem: str, state: CommunicationState, visibility: Literal["visible", "hidden"]) -> list[dict[str, str]]:
        eligible_recipient_ids = PromptComposer._eligible_recipients(actor_id, state)
        sections = [
            f"You are {actor_id}, an independent solver in a synchronous team step.",
            PromptComposer._action_contract(
                role="solver",
                eligible_recipient_ids=eligible_recipient_ids,
                final_answer_key="candidate_answer",
                allow_ask=state.can_send(actor_id),
            ),
        ]
        sections.append(f"Remaining steps: {state.max_turns - state.actor_step_count(actor_id)}.")
        if visibility == "visible":
            sections.append(f"Remaining question allowance: {state.remaining(actor_id)}.")
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "working_record": state.public_transcript(actor_id)}, ensure_ascii=False)},
        ]

    @staticmethod
    def aggregator_step_prompt(*, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState, visibility: Literal["visible", "hidden"]) -> list[dict[str, str]]:
        eligible_recipient_ids = PromptComposer._eligible_recipients("aggregator", state)
        sections = [
            "You are the aggregator in a bounded sequential review step. All solver reports are complete.",
            PromptComposer._action_contract(
                role="aggregator",
                eligible_recipient_ids=eligible_recipient_ids,
                final_answer_key="final_answer",
                allow_ask=state.can_send("aggregator"),
                requires_decision_provenance=True,
            ),
            (
                f"For a submit, valid solver IDs are {list(state.solver_ids)}. Use select_candidate only when final_answer exactly copies the selected solver's candidate answer. "
                "Use single_source_repair only when correcting one selected solver's candidate; source_solver_ids must contain exactly that solver. "
                "Use rederive_from_trace only when deriving a new answer from one or more reports or message trajectories; list every solver materially used and set selected_candidate_solver_id to null."
            ),
        ]
        sections.append(f"Remaining steps: {state.aggregator_max_steps - state.actor_step_count('aggregator')}.")
        if visibility == "visible":
            sections.append(f"Remaining question allowance: {state.remaining('aggregator')}.")
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "solver_reports": solver_reports, "review_events": state.public_transcript("aggregator")}, ensure_ascii=False)},
        ]

    @staticmethod
    def legacy_solver_step_prompt(*, actor_id: str, problem: str, state: CommunicationState, visibility: Literal["visible", "hidden"]) -> list[dict[str, str]]:
        sections = [
            f"You are {actor_id}, an independent solver in a synchronous team step.",
            "Explore the problem divergently and record a concise reasoning_note.",
            "If you ask, ask another solver about the point you are least certain about: an assumption, derivation, counterexample, or fact.",
            "You may share a candidate answer when it helps another solver verify or refute it; include the key evidence, derivation, or uncertainty rather than sending a bare answer.",
            "Your entire response must be exactly one valid JSON object: start with `{` and end with `}`.",
            "Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys.",
            'Choose exactly one of these valid forms: {"action":"continue_reasoning","reasoning_note":"brief private progress"}; '
            '{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"solver_2","question":"focused question"}; '
            '{"action":"submit","reasoning_note":"final concise rationale","candidate_answer":"exact candidate answer"}.',
        ]
        if visibility == "visible":
            sections.append(f"Remaining question allowance: {state.remaining(actor_id)}.")
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "working_record": state.public_transcript(actor_id)}, ensure_ascii=False)},
        ]

    @staticmethod
    def legacy_aggregator_action_prompt(*, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState, visibility: Literal["visible", "hidden"]) -> list[dict[str, str]]:
        sections = [
            "You are the aggregator. All solver reports are complete.",
            "Decide whether one focused clarification is necessary before final synthesis.",
            "Your entire response must be exactly one valid JSON object: start with `{` and end with `}`.",
            "Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys.",
            'Choose exactly one of these valid forms: {"action":"finalize"}; '
            '{"action":"ask","recipient_id":"solver_1","question":"one focused clarification"}.',
        ]
        if visibility == "visible":
            sections.append(f"Remaining question allowance: {state.remaining('aggregator')}.")
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "solver_reports": solver_reports, "review_events": state.public_transcript("aggregator")}, ensure_ascii=False)},
        ]

    @staticmethod
    def single_step_prompt(*, problem: str, reasoning_notes: list[str]) -> list[dict[str, str]]:
        return [{"role": "system", "content": "You are an independent solver completing a private reasoning step. Explore the problem carefully and record a concise reasoning_note. Your entire response must be exactly one valid JSON object: start with `{` and end with `}`. Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys. Choose exactly one: {\"action\":\"continue_reasoning\",\"reasoning_note\":\"brief private progress\"}; {\"action\":\"submit\",\"reasoning_note\":\"final concise rationale\",\"candidate_answer\":\"exact candidate answer\"}."}, {"role": "user", "content": json.dumps({"problem": problem, "reasoning_notes": reasoning_notes}, ensure_ascii=False)}]

    @staticmethod
    def single_final_prompt(*, problem: str, reasoning_notes: list[str]) -> list[dict[str, str]]:
        return [{"role": "system", "content": "You are making a forced final submission after private reasoning steps. Return exactly one XML element and no other text: <final_answer>the exact answer required by the task</final_answer>. Inside it include only the final answer value, with no reasoning, Markdown, labels, or explanation."}, {"role": "user", "content": json.dumps({"problem": problem, "reasoning_notes": reasoning_notes}, ensure_ascii=False)}]

    @staticmethod
    def solver_reply_prompt(*, actor_id: str, problem: str, question: dict[str, object], state: CommunicationState) -> list[dict[str, str]]:
        return [{"role": "system", "content": f"You are {actor_id}. Answer the received question concisely using your working record. Return plain reply text only."}, {"role": "user", "content": json.dumps({"problem": problem, "question": question, "working_record": state.public_transcript(actor_id)}, ensure_ascii=False)}]

    @staticmethod
    def json_repair_prompt(
        *,
        invalid_output: str,
        action_kind: Literal["solver", "aggregator", "legacy_aggregator"],
        solver_ids: tuple[str, ...] = (),
    ) -> list[dict[str, str]]:
        if action_kind == "legacy_aggregator":
            examples = 'Choose exactly one: {"action":"finalize"}; {"action":"ask","recipient_id":"solver_1","question":"one focused clarification"}.'
        elif action_kind == "solver":
            examples = 'Choose exactly one: {"action":"think","reasoning_note":"brief private progress"}; {"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}; {"action":"submit","reasoning_note":"final concise rationale","candidate_answer":"exact candidate answer"}.'
        else:
            examples = (
                'Choose exactly one: {"action":"think","reasoning_note":"brief review progress"}; '
                '{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}; '
                '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"exact candidate answer","decision_mode":"select_candidate","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":"<solver_id>"}; '
                '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"corrected exact answer","decision_mode":"single_source_repair","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":"<solver_id>"}; '
                '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"new exact answer","decision_mode":"rederive_from_trace","source_solver_ids":["<solver_id>"],"selected_candidate_solver_id":null}. '
                f"For an aggregator submit, valid solver IDs are {list(solver_ids)}. select_candidate copies the selected candidate exactly; single_source_repair uses exactly one source; rederive_from_trace lists every material source and uses null selected_candidate_solver_id."
            )
        return [{"role": "system", "content": "Your previous response was not valid for the required action schema. Re-output only valid JSON, with no Markdown fences or explanation. Your entire response must start with `{` and end with `}`. " + examples}, {"role": "user", "content": invalid_output}]

    @staticmethod
    def final_aggregator_prompt(*, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState) -> list[dict[str, str]]:
        return [{"role": "system", "content": "You are the forced final aggregator. First reason internally over the reports and review events. Choose exactly one: select_candidate (copy one solver candidate unchanged), single_source_repair (correct one solver's candidate), or rederive_from_trace (derive a new answer from one or more trajectories). Return exactly two adjacent XML elements and no other text. First: <aggregation_decision>{\"mode\":\"select_candidate\", \"single_source_repair\", or \"rederive_from_trace\",\"source_solver_ids\":[\"<solver_id>\"],\"selected_candidate_solver_id\":\"<solver_id>\" or null,\"rationale\":\"brief evidence-based reason\"}</aggregation_decision>. For select_candidate, selected_candidate_solver_id is required, must occur in source_solver_ids, and final_answer must exactly copy that solver candidate. For single_source_repair, source_solver_ids must contain exactly selected_candidate_solver_id. For rederive_from_trace, source_solver_ids must name every solver report or message trajectory materially used and selected_candidate_solver_id must be null. Second: <final_answer>the exact answer required by the task</final_answer>. Inside final_answer, include only the answer value in the exact form requested by the problem. Do not include reasoning, Markdown, labels, explanations, or restate the question. For a binary question, output exactly `Yes` or `No`. For a number, output only the number; never add words such as `Position` or `Answer`. For multiple-choice or list answers, preserve the exact separators and ordering requested by the problem."}, {"role": "user", "content": json.dumps({"problem": problem, "solver_reports": solver_reports, "review_events": state.public_transcript("aggregator")}, ensure_ascii=False)}]

    @staticmethod
    def final_aggregation_repair_prompt(*, invalid_output: str, state: CommunicationState) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Your previous forced-final response had invalid aggregation provenance. Re-output exactly two adjacent XML elements and no other text. "
                    "First: <aggregation_decision>{valid JSON}</aggregation_decision>. Second: <final_answer>exact answer</final_answer>. "
                    f"Valid solver IDs are {list(state.solver_ids)}. mode is select_candidate, single_source_repair, or rederive_from_trace. "
                    "select_candidate requires selected_candidate_solver_id in source_solver_ids; single_source_repair requires exactly one source equal to selected_candidate_solver_id; "
                    "rederive_from_trace requires selected_candidate_solver_id null."
                ),
            },
            {"role": "user", "content": invalid_output},
        ]

    @staticmethod
    def legacy_final_aggregator_prompt(*, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState) -> list[dict[str, str]]:
        return PromptComposer.final_aggregator_prompt(problem=problem, solver_reports=solver_reports, state=state)
