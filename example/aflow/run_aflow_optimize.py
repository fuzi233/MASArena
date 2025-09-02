import argparse
import os
from dotenv import load_dotenv
from typing import Dict, Any

from mas_arena.agents import AgentSystem, AgentSystemRegistry
from mas_arena.evaluators import BENCHMARKS
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.optimizers.aflow.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG

load_dotenv()

def get_config_from_env() -> Dict[str, Any]:
    """Loads configuration from environment variables."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_api_base": os.getenv("OPENAI_API_BASE"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "anthropic_api_base": os.getenv("ANTHROPIC_API_BASE") or os.getenv("OPENAI_API_BASE"),
        "optimizer_model_name": os.getenv("OPTIMIZER_MODEL_NAME", "gpt-4o"),
        "executor_model_name": os.getenv("EXECUTOR_MODEL_NAME") or os.getenv("MODEL_NAME", "gpt-4o-mini"),
    }

def initialize_agent(agent_type: str, config: Dict[str, Any]) -> AgentSystem:
    """Initializes an agent system based on the specified type and configuration."""
    if agent_type == "optimizer":
        agent_config = {
            "model_name": config["optimizer_model_name"],
            "api_key": config["anthropic_api_key"],
            "api_base": config["anthropic_api_base"],
        }
    elif agent_type == "executor":
        agent_config = {
            "model_name": config["executor_model_name"],
            "api_key": config["openai_api_key"],
            "api_base": config["openai_api_base"],
        }
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    agent = AgentSystemRegistry.get("single_agent", agent_config)
    if agent is None:
        raise ValueError(f"Failed to initialize {agent_type} agent.")
    return agent


def get_evaluator(benchmark_name: str) -> BaseEvaluator:
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}. Available: {list(BENCHMARKS.keys())}")
    evaluator_class = BENCHMARKS[benchmark_name]["evaluator"]
    return evaluator_class(benchmark_name, {})


def run_aflow_optimization(args: argparse.Namespace) -> str:
    """
    Runs the AFlow optimization process and returns the path to the optimized graph.
    """
    # Initialization
    env_config = get_config_from_env()
    optimizer_agent = initialize_agent("optimizer", env_config)
    executor_agent = initialize_agent("executor", env_config)
    evaluator = get_evaluator(args.benchmark)

    # Create and run optimizer
    optimizer = AFlowOptimizer(
        graph_path=args.graph_path,
        optimized_path=args.optimized_path,
        optimizer_agent=optimizer_agent,
        executor_agent=executor_agent,
        validation_rounds=args.validation_rounds,
        eval_rounds=args.eval_rounds,
        max_rounds=args.max_rounds,
        **EXPERIMENTAL_CONFIG.get(args.benchmark, {}),
    )

    optimizer.setup()
    optimizer.optimize(evaluator)
    
    final_graph_path = os.path.join(args.optimized_path, "final_graph.json")
    print(f"\n[AFlow] Optimization complete. Optimized graph saved to: {final_graph_path}")
    return final_graph_path


def main():
    """Main function to run the AFlow optimization process."""
    parser = argparse.ArgumentParser(description="Run AFlow optimization")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=list(["humaneval"]),
        help="Benchmark to run.",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="mas_arena/configs/aflow",
        help="Path to the agent flow graph configuration.",
    )
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="example/aflow/humaneval/optimization",
        help="Path to save the optimized agent flow graph.",
    )
    parser.add_argument("--validation_rounds", type=int, default=1, help="Number of validation rounds.")
    parser.add_argument("--eval_rounds", type=int, default=1, help="Number of evaluation rounds.")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of optimization rounds.")
    args = parser.parse_args()

    optimized_graph_path = run_aflow_optimization(args)
    
    print("\n" + "=" * 80)
    print("Standalone AFlow optimization process finished.")
    print(f"The optimized agent graph is saved at: {optimized_graph_path}")
    print("\nTo evaluate this optimized agent, run the benchmark using the main runner:")
    print(f"python main.py --benchmark {args.benchmark} --agent-system single_agent --agent-graph-config {optimized_graph_path} --limit 10")
    print("=" * 80)


if __name__ == "__main__":
    main()
