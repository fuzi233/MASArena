#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Runner

This module provides functionality for running benchmarks on agent systems.
"""

import os
import json
import random
import re
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from openai.types.completion_usage import CompletionUsage
import traceback
from rich import print as rprint

from mas_arena.metrics import (
    MetricsRegistry,
    MetricsCollector
)
from mas_arena.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.utils.normalization import normalize_problem_keys

def custom_json_serializer(obj):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if hasattr(obj, '__dict__'):
        # For AIMessage-like objects, convert to dict
        return {key: getattr(obj, key) for key in obj.__dict__ if not key.startswith('_')}
    if isinstance(obj, CompletionUsage):
        return {
            "prompt_tokens": obj.prompt_tokens,
            "completion_tokens": obj.completion_tokens,
            "total_tokens": obj.total_tokens,
        }
    try:
        return str(obj)  # Fallback for other non-serializable types
    except Exception:
        return f"Object of type {type(obj).__name__} is not JSON serializable"

class BenchmarkRunner:
    """
    Simple interface for running benchmarks on multi-agent systems.

    Examples:
        >>> benchmark = BenchmarkRunner()
        >>> results = benchmark.run("math", limit=5)
    """

    def __init__(self, results_dir="results"):
        """
        Initialize the benchmark runner.

        Args:
            results_dir: Directory to save results
            metrics_dir: Directory to save metrics data
        """
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.agent_config = None  # Store agent configuration
        self.experiment_config = {}
        self.metrics_registry = None
        self.metrics_collector = None

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        # os.makedirs(metrics_dir, exist_ok=True)

        # Set up metrics
        self.metrics_registry = self._setup_metrics()
        
        # Create centralized metrics collector
        self.metrics_collector = MetricsCollector(self.metrics_registry)

    @staticmethod
    def _agent_label(agent_system, agent_config):
        """Return a filesystem-safe, concise label for experiment outputs."""
        if agent_system == "communication_budget":
            legacy_budget = agent_config.get("communication_budget")
            solver_budget = agent_config.get("solver_communication_budget")
            if solver_budget is None:
                solver_budget = legacy_budget if legacy_budget is not None else 1
            aggregator_budget = agent_config.get("aggregator_communication_budget")
            if aggregator_budget is None:
                aggregator_budget = legacy_budget if legacy_budget is not None else solver_budget
            aggregator_steps = agent_config.get("aggregator_max_steps") or agent_config.get("max_turns", 4)
            protocol = agent_config.get("protocol_version", "bounded-v2")
            raw = (
                f"communication_budget_n{agent_config.get('solver_count', 2)}"
                f"_sk{solver_budget}_ak{aggregator_budget}"
                f"_{agent_config.get('budget_visibility', 'visible')}"
                f"_t{agent_config.get('max_turns', 4)}_at{aggregator_steps}_p{protocol}"
                f"_temp{agent_config.get('temperature', 1.0)}"
            )
        elif agent_system == "step_single_agent":
            raw = f"step_single_agent_t{agent_config.get('max_turns', 4)}_temp{agent_config.get('temperature', 1.0)}"
        else:
            raw = agent_system
        return re.sub(r"[^A-Za-z0-9_-]+", "", raw.replace(".", "p"))

    @staticmethod
    def _summary_agent_config(agent_config):
        """Preserve reproducibility settings without serializing clients or secrets."""
        blocked = {"api_key", "openai_api_key", "model_client"}
        return {
            key: value
            for key, value in agent_config.items()
            if key.lower() not in blocked and isinstance(value, (str, int, float, bool, type(None), list, dict))
        }

    def _setup_metrics(self):
        """Set up metrics collection"""
        registry = MetricsRegistry()
        return registry

    @staticmethod
    def _communication_stats(team_state):
        """Derive compact aggregate counts while retaining the raw trace separately."""
        events = team_state.get("events", []) if isinstance(team_state, dict) else []
        action_kind = {
            "continue_reasoning": "think",
            "aggregator_think": "think",
            "solver_question": "ask",
            "aggregator_question": "ask",
            "solver_submission": "submit",
            "aggregator_submission": "submit",
            "json_repair": "json_repair",
            "aggregator_provenance_repair": "provenance_repair",
            "aggregator_final_decision_repair": "provenance_repair",
            "forced_final": "forced_final",
        }
        role_action_counts = {}
        for event in events:
            action = action_kind.get(event.get("kind"))
            actor = event.get("sender_id")
            if action is None or not isinstance(actor, str):
                continue
            counts = role_action_counts.setdefault(
                actor,
                {"think": 0, "ask": 0, "submit": 0, "json_repair": 0, "provenance_repair": 0, "forced_final": 0},
            )
            counts[action] += 1
        return {
            "solver_questions": sum(event.get("kind") == "solver_question" for event in events),
            "aggregator_questions": sum(event.get("kind") == "aggregator_question" for event in events),
            "solver_replies": sum(event.get("kind") == "solver_reply" for event in events),
            "aggregator_replies": sum(event.get("kind") == "aggregator_reply" for event in events),
            "charged_question_budget": sum(int(event.get("charged_budget", 0)) for event in events),
            "solver_steps": team_state.get("solver_steps", {}),
            "role_action_counts": role_action_counts,
        }

    def _prepare_benchmark(self, benchmark_name, data_path, limit, agent_system, agent_config, verbose, data_id=None, seed=42):
        """
        Run a benchmark with the specified configuration.

        Args:
            benchmark_name: Name of the benchmark (math, drop, gsm8k, hotpotqa, humaneval, mbpp)
            data_path: Custom path to benchmark data file (optional)
            limit: Maximum number of problems to process
            agent_system: Agent system to use (single_agent, supervisor_mas, swarm)
            agent_config: Configuration for the agent system
            verbose: Whether to print progress information

        Returns:
            Dictionary of benchmark results
        """
        # Validate benchmark name
        if benchmark_name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Supported: {', '.join(BENCHMARKS.keys())}")
        
        benchmark_config = BENCHMARKS[benchmark_name]

        if verbose:
                print(f"Available agent systems: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")
        self.agent_config = agent_config or {}

        # Ensure the agent knows which evaluator to use by setting it in the config
        self.agent_config['evaluator'] = benchmark_name

        if not data_path:
            data_path = benchmark_config.get("data_path", f"data/{benchmark_name}_test.jsonl")

        output_file = Path(self.results_dir) / f"{benchmark_name}_{self._agent_label(agent_system, self.agent_config)}_seed{seed}_{self.timestamp}.json"

        agent = create_agent_system(agent_system, self.agent_config)
        if agent is None:
            raise ValueError(f"Unknown agent system: {agent_system}. Available: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")

        agent.set_metrics_registry(self.metrics_registry)

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                problems = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_id:
            primary_id = benchmark_config.get("normalization_keys", {}).get("id", None)
            if primary_id is not None:
                for problem in problems:
                    if str(problem[primary_id]) == data_id:
                        problems = [problem]
                        break


        if limit and limit < len(problems):
            problems = random.Random(seed).sample(problems, limit)

        return agent, problems, benchmark_config, output_file

    async def _process_one_problem(self, i, p, agent, benchmark_config, verbose=True):
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(p, key_mapping, i)
        problem_id = normalized_problem["id"]

        if verbose:
            print(f"Problem {i + 1}: {problem_id}")

        self.metrics_collector.start_timer(f"mas_arena.problem.{problem_id}", {"problem_id": problem_id})

        try:
            results = await agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)
            problem_duration_ms = self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")

            duration_ms = results.get("execution_time_ms", problem_duration_ms)
            score = results.get("score", 0)
            is_correct = results.get("is_correct", score == 1)

            self.metrics_collector.record_metric("mas_arena.problem.result", score, {"problem_id": problem_id, "correct": is_correct, "duration_ms": duration_ms})

            result_entry = {
                "problem_id": problem_id,
                "problem": normalized_problem["problem"],
                "expected": normalized_problem["solution"],
                "prediction": results.get("extracted_answer", ""),
                "score": score,
                "is_correct": is_correct,
                "status": results.get("status"),
                "reasoning": results.get("reasoning", ""),
                "duration_ms": duration_ms,
                "agent_system": agent.name,
                "llm_usage": results.get("llm_usage", {}),
                "summary": {"correct": is_correct, "score": score, "duration_ms": duration_ms},
            }
            if results.get("status") == "error":
                result_entry["error"] = results.get("error", results.get("reasoning", ""))
                result_entry["error_type"] = results.get("error_type", "UnknownError")
            team_state = results.get("team_state")
            if isinstance(team_state, dict):
                result_entry["communication_trace"] = team_state
                result_entry["communication_stats"] = self._communication_stats(team_state)
                result_entry["aggregation_decision"] = team_state.get("aggregation_decision")
            if verbose:
                status_char = "E" if results.get("status") == "error" else "✓" if is_correct else "✗"
                error_summary = ""
                if status_char == "E":
                    raw_error = str(results.get("error", results.get("reasoning", "")))
                    error_summary = f" — {' '.join(raw_error.split())[:500]}" if raw_error else ""
                print(f"Result: {status_char} ({duration_ms:.0f}ms){error_summary}")
            return result_entry
        except Exception as e:
            self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")
            self.metrics_collector.record_error("problem_processing", str(e), {"problem_id": problem_id, "error_type": type(e).__name__})
            if verbose:
                print(f"Error processing problem {problem_id}: {e}")
                traceback.print_exc()
            return {"problem_id": problem_id, "problem": normalized_problem.get("problem"), "status": "error", "error": str(e), "score": 0}

    def _finalize_benchmark(self, all_results, benchmark_name, agent_system, output_file, verbose):
        total = len(all_results)
        if total == 0:
            print("No results to finalize.")
            return {}

        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        errored = sum(1 for r in all_results if r.get("status") == "error")
        
        # Calculate accuracy only on non-errored problems
        valid_runs = total - errored
        accuracy = correct / valid_runs if valid_runs > 0 else 0

        total_duration = sum(r.get("duration_ms", 0) for r in all_results)
        
        # Report all consumed tokens, including requests made before a failed problem.
        completed_runs = [r for r in all_results if r.get("status") != "error"]
        total_tokens_all = sum(r.get("llm_usage", {}).get("total_tokens", 0) for r in all_results)
        total_tokens_completed = sum(r.get("llm_usage", {}).get("total_tokens", 0) for r in completed_runs)
        avg_tokens_all = total_tokens_all / total if total > 0 else 0
        avg_tokens_completed = total_tokens_completed / len(completed_runs) if completed_runs else 0

        summary = {
            "benchmark": benchmark_name,
            "agent_system": agent_system,
            "total_problems": total,
            "correct": correct,
            "errored": errored,
            "accuracy": accuracy,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / total if total > 0 else 0,
            "total_tokens_all_problems": total_tokens_all,
            "avg_tokens_per_problem": avg_tokens_all,
            "total_tokens_completed_problems": total_tokens_completed,
            "avg_tokens_per_completed_problem": avg_tokens_completed,
            # Backward-compatible alias. Historically this meant non-errored, not correct, problems.
            "avg_tokens_per_successful_problem": avg_tokens_completed,
            "results_file": str(output_file),
            "experiment_config": self.experiment_config,
            # "metrics_dir": str(metrics_output),
            "timestamp": self.timestamp,
        }

        self.metrics_registry.stop_all_collectors()
        # self.metrics_registry.export_all(format="json", path=str(metrics_output))

        # Create a separate summary file for visualization
        summary_file = output_file.with_name(f"{output_file.stem}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4, default=custom_json_serializer)

        # Save main results file
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "results": all_results}, f, indent=4, default=custom_json_serializer)
       
        if verbose:
            # Run failure attribution analysis for incorrect results
            self._run_failure_attribution(all_results, agent_system, verbose)
            # Print benchmark summary
            print("\n" + "=" * 80)
            rprint("[bold green]🎯 Benchmark Summary[/bold green]")
            print("=" * 80)
            print(json.dumps(summary, indent=2))
            print("-" * 80)
            # Print visualization command
            rprint("[bold blue]📈 To visualize results, run:[/bold blue]")
            print(f"$ python mas_arena/visualization/visualize_benchmark.py visualize --summary {summary_file}")
            print("=" * 80)


        self.results = all_results
        self.summary = summary
        return summary

    def _collect_failed_responses(self, all_results, agent_system, verbose):
        """
        Collect and move failed agent response files to a centralized directory.
        
        Args:
            all_results: List of all benchmark results
            agent_system: Name of the agent system used
            verbose: Whether to print progress information
            
        Returns:
            Path to the directory containing failed responses, or None if no failures
        """
        # Find incorrect results
        incorrect_results = [r for r in all_results if r.get("score", 0) != 1 and "error" not in r]
        
        if not incorrect_results:
            return None
        
        # Find corresponding agent response files
        agent_responses_dir = Path("results/agent_responses")
        if not agent_responses_dir.exists():
            if verbose:
                print("Agent responses directory not found.")
            return None
        
        # Create centralized failed responses directory
        failed_responses_dir = Path(self.results_dir) / "failure" / f"failed_responses_{self.timestamp}"
        failed_responses_dir.mkdir(parents=True, exist_ok=True)
        
        collected_files = []
        
        for result in incorrect_results:
            problem_id = result.get("problem_id")
            if not problem_id:
                continue
                
            # Find the corresponding agent response file
            # Pattern: {agent_system}_{problem_id}_{timestamp}_{hash}.json
            pattern = f"{agent_system}_{problem_id}_*.json"
            matching_files = list(agent_responses_dir.glob(pattern))
            
            if not matching_files:
                if verbose:
                    print(f"No agent response file found for problem {problem_id}")
                continue
            
            # Use the most recent file if multiple matches
            agent_response_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            
            # Copy file to failed responses directory
            dest_file = failed_responses_dir / agent_response_file.name
            shutil.copy2(agent_response_file, dest_file)
            collected_files.append(dest_file)
            
            if verbose:
                print(f"Collected failed response for problem {problem_id}: {agent_response_file.name}")
        
        if verbose:
            print(f"\nCollected {len(collected_files)} failed response files in: {failed_responses_dir}")
        
        return failed_responses_dir

    def _run_failure_attribution(self, all_results, agent_system, verbose):
        """
        Prepare and display instructions for failure attribution analysis.
        
        Args:
            all_results: List of all benchmark results
            agent_system: Name of the agent system used
            verbose: Whether to print progress information
        """
        # Find incorrect results
        incorrect_results = [r for r in all_results if r.get("score", 0) != 1 and "error" not in r]
        
        if not incorrect_results:
            if verbose:
                print("\n" + "=" * 80)
                rprint("[yellow]No incorrect results found. Skipping failure attribution analysis.[/yellow]")
                print("=" * 80)
            return
        
        if verbose:
            print("\n" + "=" * 80)
            rprint(f"[bold yellow]Found {len(incorrect_results)} incorrect results for Failure Attribution Analysis[/bold yellow]")
            print("=" * 80)
        
        # Collect failed response files
        failed_responses_dir = self._collect_failed_responses(all_results, agent_system, verbose)
        
        if not failed_responses_dir:
            if verbose:
                print("Could not collect failed response files.")
            return
        
        # Check if failure inference script exists
        failure_inference_script = Path("mas_arena/failure/inference.py")
        if not failure_inference_script.exists():
            if verbose:
                print("Failure inference script not found.")
            return
        
        # Create failure output directory
        failure_output_dir = Path(self.results_dir) / "failure"
        failure_output_dir.mkdir(exist_ok=True)
        
        if verbose:
            print("\n" + "-" * 80)
            rprint("[bold blue]🔍 To run Failure Attribution Analysis, execute the following command:[/bold blue]")
            print("-" * 80)
            print(f"python {failure_inference_script} \\")
            print(f"    --method binary_search \\")
            print(f"    --model gpt-4.1 \\")
            print(f"    --directory_path {failed_responses_dir} \\")
            print(f"    --output_dir {failure_output_dir}")
            # print("-" * 80)
            rprint("\n[bold]Alternative analysis methods:[/bold]")
            print(f"#\n For comprehensive analysis:")
            print(f"python {failure_inference_script} --method all_at_once --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}")
            print(f"#\n For efficient error localization in long conversations:")
            print(f"python {failure_inference_script} --method binary_search --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}")
            print(f"\n# For detailed incremental analysis:")
            print(f"python {failure_inference_script} --method step_by_step --model gpt-4.1 --directory_path {failed_responses_dir} --output_dir {failure_output_dir}")

            print("=" * 80)

    def run(self, benchmark_name="math", data_path=None, limit=None, agent_system="single_agent", agent_config=None, verbose=True, data_id=None, seed=42):
        """
        Run a benchmark sequentially. This is a wrapper around arun.
        """
        return asyncio.run(self.arun(
            benchmark_name=benchmark_name,
            data_path=data_path,
            limit=limit,
            agent_system=agent_system,
            agent_config=agent_config,
            verbose=verbose,
            data_id=data_id,
            seed=seed,
            concurrency=1  # Run sequentially
        ))

    async def arun(self, benchmark_name="math", data_path=None, limit=None, agent_system="single_agent", agent_config=None, verbose=True, data_id=None, concurrency=10, seed=42):
        # Prepare benchmark; we only need problems and config here
        _, problems, benchmark_config, output_file = self._prepare_benchmark(
            benchmark_name, data_path, limit, agent_system, agent_config, verbose, data_id, seed
        )
        self.experiment_config = {
            "agent_config": self._summary_agent_config(self.agent_config),
            "data_path": str(data_path or benchmark_config.get("data_path")),
            "requested_limit": limit,
            "actual_problem_count": len(problems),
            "concurrency": concurrency,
            "seed": seed,
            "selected_problem_ids": [str(problem.get(benchmark_config.get("normalization_keys", {}).get("id", "id"), "")) for problem in problems],
        }

        if verbose:
            print(f"Running {benchmark_name} benchmark asynchronously with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems with concurrency {concurrency}.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")

        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(i, p):
            async with semaphore:
                # Create a fresh agent instance per problem to isolate state
                new_agent = create_agent_system(agent_system, self.agent_config)
                new_agent.set_metrics_registry(self.metrics_registry)
                try:
                    return await self._process_one_problem(i, p, new_agent, benchmark_config, verbose)
                finally:
                    close = getattr(new_agent, "aclose", None)
                    if callable(close):
                        await close()

        tasks = [process_with_semaphore(i, p) for i, p in enumerate(problems)]
        
        all_results = await tqdm.gather(*tasks, desc="Processing Problems")

        return self._finalize_benchmark(all_results, benchmark_name, agent_system, output_file, verbose)

    def visualize_results(self, output_dir=None):
        """
        Generate visualizations from the benchmark results.

        Args:
            output_dir: Directory to save visualizations (defaults to metrics_dir/benchmark_timestamp/viz)
        """
        pass
