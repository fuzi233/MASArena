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
    def _action_contract(*, role: str, eligible_recipient_ids: tuple[str, ...], final_answer_key: str) -> str:
        return (
            f"You are the {role}. Explore the problem and record a concise reasoning_note. "
            "If you ask, ask one eligible solver about the assumption, derivation, counterexample, or fact you are least certain about. "
            "Do not repeat a question already present in your working record. "
            f"eligible_recipient_ids (ordered only for display): {list(eligible_recipient_ids)}. Choose by relevance, never by identifier order. "
            "Your entire response must be exactly one valid JSON object: start with `{` and end with `}`. "
            "Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys. "
            'Choose exactly one valid form: {"action":"think","reasoning_note":"brief private progress"}; '
            '{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}; '
            f'{{"action":"submit","reasoning_note":"final concise rationale","{final_answer_key}":"exact answer"}}.'
        )

    @staticmethod
    def solver_step_prompt(*, actor_id: str, problem: str, state: CommunicationState, visibility: Literal["visible", "hidden"]) -> list[dict[str, str]]:
        eligible_recipient_ids = PromptComposer._eligible_recipients(actor_id, state)
        sections = [
            f"You are {actor_id}, an independent solver in a synchronous team step.",
            PromptComposer._action_contract(
                role="solver", eligible_recipient_ids=eligible_recipient_ids, final_answer_key="candidate_answer"
            ),
        ]
        if visibility == "visible":
            sections.append(
                f"Remaining question allowance: {state.remaining(actor_id)}. Remaining steps: {state.max_turns - state.actor_step_count(actor_id)}."
            )
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "working_record": state.public_transcript(actor_id)}, ensure_ascii=False)},
        ]

    @staticmethod
    def aggregator_step_prompt(
        *, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState, visibility: Literal["visible", "hidden"]
    ) -> list[dict[str, str]]:
        eligible_recipient_ids = PromptComposer._eligible_recipients("aggregator", state)
        sections = [
            "You are the aggregator in a bounded sequential review step. All solver reports are complete.",
            PromptComposer._action_contract(
                role="aggregator", eligible_recipient_ids=eligible_recipient_ids, final_answer_key="final_answer"
            ),
        ]
        if visibility == "visible":
            sections.append(
                f"Remaining question allowance: {state.remaining('aggregator')}. Remaining steps: {state.aggregator_max_steps - state.actor_step_count('aggregator')}."
            )
        return [
            {"role": "system", "content": "\n".join(sections)},
            {"role": "user", "content": json.dumps({"problem": problem, "solver_reports": solver_reports, "review_events": state.public_transcript("aggregator")}, ensure_ascii=False)},
        ]

    @staticmethod
    def single_step_prompt(*, problem: str, reasoning_notes: list[str]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are an independent solver completing a private reasoning step. Explore the problem carefully and record a concise reasoning_note. Your entire response must be exactly one valid JSON object: start with `{` and end with `}`. Do not use Markdown fences (```), prose before or after the JSON, comments, or additional keys. Choose exactly one: {\"action\":\"continue_reasoning\",\"reasoning_note\":\"brief private progress\"}; {\"action\":\"submit\",\"reasoning_note\":\"final concise rationale\",\"candidate_answer\":\"exact candidate answer\"}."},
            {"role": "user", "content": json.dumps({"problem": problem, "reasoning_notes": reasoning_notes}, ensure_ascii=False)},
        ]

    @staticmethod
    def single_final_prompt(*, problem: str, reasoning_notes: list[str]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are making a forced final submission after private reasoning steps. Return exactly one XML element and no other text: <final_answer>the exact answer required by the task</final_answer>. Inside it include only the final answer value, with no reasoning, Markdown, labels, or explanation."},
            {"role": "user", "content": json.dumps({"problem": problem, "reasoning_notes": reasoning_notes}, ensure_ascii=False)},
        ]

    @staticmethod
    def solver_reply_prompt(*, actor_id: str, problem: str, question: dict[str, object], state: CommunicationState) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"You are {actor_id}. Answer the received question concisely using your working record. Return plain reply text only."},
            {"role": "user", "content": json.dumps({"problem": problem, "question": question, "working_record": state.public_transcript(actor_id)}, ensure_ascii=False)},
        ]

    @staticmethod
    def json_repair_prompt(*, invalid_output: str, action_kind: Literal["solver", "aggregator"]) -> list[dict[str, str]]:
        examples = (
            'Choose exactly one: {"action":"think","reasoning_note":"brief private progress"}; '
            '{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}; '
            '{"action":"submit","reasoning_note":"final concise rationale","candidate_answer":"exact candidate answer"}.'
            if action_kind == "solver"
            else 'Choose exactly one: {"action":"think","reasoning_note":"brief review progress"}; '
            '{"action":"ask","reasoning_note":"specific uncertainty","recipient_id":"<eligible_solver_id>","question":"focused question"}; '
            '{"action":"submit","reasoning_note":"final concise rationale","final_answer":"exact answer"}.'
        )
        return [
            {
                "role": "system",
                "content": (
                    "Your previous response was not valid for the required action schema. "
                    "Re-output only valid JSON, with no Markdown fences or explanation. "
                    "Your entire response must start with `{` and end with `}`. "
                    f"{examples}"
                ),
            },
            {"role": "user", "content": invalid_output},
        ]

    @staticmethod
    def final_aggregator_prompt(*, problem: str, solver_reports: list[dict[str, object]], state: CommunicationState) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are the forced final aggregator. First reason internally over the reports and review events. Either select one solver candidate, or rederive the answer from the trajectory. Return exactly two adjacent XML elements and no other text. First: <aggregation_decision>{\"mode\":\"select_candidate\" or \"rederive_from_trace\",\"source_solver_ids\":[\"<solver_id>\"],\"selected_candidate_solver_id\":\"<solver_id>\" or null,\"rationale\":\"brief evidence-based reason\"}</aggregation_decision>. For select_candidate, selected_candidate_solver_id is required. For rederive_from_trace, source_solver_ids must name every solver report or message trajectory materially used. Second: <final_answer>the exact answer required by the task</final_answer>. Inside final_answer, include only the answer value in the exact form requested by the problem. Do not include reasoning, Markdown, labels, explanations, or restate the question. For a binary question, output exactly `Yes` or `No`. For a number, output only the number; never add words such as `Position` or `Answer`. For multiple-choice or list answers, preserve the exact separators and ordering requested by the problem."},
            {"role": "user", "content": json.dumps({"problem": problem, "solver_reports": solver_reports, "review_events": state.public_transcript("aggregator")}, ensure_ascii=False)},
        ]
