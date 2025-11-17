import os
import glob
import yaml
from typing import List, Dict, Any, Tuple
import asyncio

from mas_arena.agents import create_agent_system
from mas_arena.evaluators.ALFWorld.environment import AlfredThorEnv
from ..registry import register_benchmark
from ..base_evaluator import BaseEvaluator

# Constants
MAX_STEPS = 100

@register_benchmark("alfworld", normalization_keys={})
class AlfWorldEvaluator(BaseEvaluator):
    """
    Evaluator for the ALFWorld benchmark.
    This benchmark runs tasks in a simulated environment and does not use a standard
    dataset of problems. Instead, it discovers task files to execute.
    """
    SUPPORTS_CONCURRENCY = False  # ALFWorld environment cannot be run in parallel

    def __init__(self, name: str = "alfworld", config: Dict[str, Any] = None):
        """
        Custom initializer for AlfWorldEvaluator.
        It calls the parent initializer but is configured to skip standard data loading.
        """
        print("DEBUG - AlfWorldEvaluator.__init__ - Starting initialization")
        print(f"DEBUG - AlfWorldEvaluator.__init__ - Initial config: {config}")
        
        try:
            super().__init__(name, config=config)
            print(f"DEBUG - AlfWorldEvaluator.__init__ - After super().__init__, self.config: {self.config}")
            
            # Load ALFWorld-specific config and merge it into the main config
            alfworld_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            print(f"DEBUG - AlfWorldEvaluator.__init__ - Looking for config file at: {alfworld_config_path}")
            print(f"DEBUG - AlfWorldEvaluator.__init__ - Config file exists: {os.path.exists(alfworld_config_path)}")
            
            if os.path.exists(alfworld_config_path):
                try:
                    with open(alfworld_config_path, 'r') as f:
                        alfworld_specific_config = yaml.safe_load(f)
                    print(f"DEBUG - AlfWorldEvaluator.__init__ - Loaded config: {alfworld_specific_config}")
                    
                    if self.config is None:
                        self.config = {}
                        print("DEBUG - AlfWorldEvaluator.__init__ - self.config was None, initializing empty dict")
                    
                    # Merge and overwrite keys in the 'alfworld' section of the main config
                    self.config.setdefault('alfworld', {}).update(alfworld_specific_config)
                    print(f"DEBUG - AlfWorldEvaluator.__init__ - Final merged config: {self.config}")
                except Exception as e:
                    print(f"DEBUG - AlfWorldEvaluator.__init__ - Error loading config file: {e}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print("DEBUG - AlfWorldEvaluator.__init__ - Warning: config.yaml not found")
        except Exception as e:
            print(f"DEBUG - AlfWorldEvaluator.__init__ - Error during initialization: {e}")
            import traceback
            print(traceback.format_exc())

    def _load_data(self):
        """
        Override the base data loading method. For ALFWorld, tasks are discovered
        dynamically in the `run` method, so no upfront data loading is needed.
        """
        pass  # Do nothing

    async def run(self, agent_system: str, agent_config: dict, data_path: str = None, limit: int = None, verbose: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Overrides the base run method to implement ALFWorld's specific task discovery logic.
        """
        print("Starting ALFWorld evaluation...")
        
        # 调试信息：检查self.config
        print(f"DEBUG - self.config: {self.config}")

        # 1. Load configuration
        alfworld_config = self.config.get("alfworld", {}) if self.config else {}
        print(f"DEBUG - alfworld_config: {alfworld_config}")

        # 2. Discover Task Files from config
        dataset_config = alfworld_config.get('dataset', {})
        print(f"DEBUG - dataset_config: {dataset_config}")
        
        # 优先从config.yaml加载路径
        alfworld_data_path = os.path.expandvars(dataset_config.get('data_path', ''))
        split = dataset_config.get('split', 'valid_unseen')

        # 仅当config中未指定路径时，才回退到环境变量
        if not alfworld_data_path and 'ALFWORLD_DATA' in os.environ:
            print(f"DEBUG - data_path not in config, falling back to ALFWORLD_DATA environment variable: {os.environ['ALFWORLD_DATA']}")
            alfworld_data_path = os.environ['ALFWORLD_DATA']
        elif alfworld_data_path:
            print(f"DEBUG - Using data_path from config file: {alfworld_data_path}")
        else:
            print("DEBUG - ALFWORLD_DATA not found in environment and no data_path in config.")

        if not alfworld_data_path:
            print("\nError: ALFWorld data path not configured.")
            print("Please set 'data_path' under the 'dataset' key in 'mas_arena/evaluators/ALFWorld/config.yaml',")
            print("or set the ALFWORLD_DATA environment variable.")
            return {}
        
        # Verify that the path exists
        if not os.path.isdir(alfworld_data_path):
            print(f"\nError: The specified ALFWorld data path does not exist or is not a directory.")
            print(f"Path specified: {alfworld_data_path}")
            print("Please check the 'data_path' in your config file or the ALFWORLD_DATA environment variable.")
            return {}

        full_task_path = os.path.join(alfworld_data_path, split)
        
        # Verify that the split sub-directory exists
        if not os.path.isdir(full_task_path):
            print(f"\nError: The '{split}' split directory was not found in the ALFWorld data path.")
            print(f"Searched for: {full_task_path}")
            print(f"Please ensure the directory exists and contains the correct data.")
            return {}
            
        task_files = glob.glob(os.path.join(full_task_path, "**", "traj_data.json"), recursive=True)

        if not task_files:
            print(f"\nError: No ALFWorld task files ('traj_data.json') found.")
            print(f"Searched in: {full_task_path}")
            print("Please ensure the path is correct and contains the ALFWorld JSON data.")
            return {}

        if limit and limit < len(task_files):
            task_files = task_files[:limit]
            
        # 3. Create a single agent instance for the entire evaluation
        agent = create_agent_system(agent_system, agent_config)
        if not agent:
            raise ValueError(f"Could not create agent system: {agent_system}")

        # 4. Create environment
        print(f"DEBUG - Creating environment with config: {alfworld_config}")
        try:
            env = AlfredThorEnv(alfworld_config)
            print("DEBUG - Environment created successfully")
        except Exception as e:
            print(f"DEBUG - Error creating environment: {e}")
            import traceback
            print(traceback.format_exc())
            return {"error": f"Failed to create environment: {str(e)}"}

        # 5. Run evaluation loop for all tasks
        all_results = []
        for i, task_file in enumerate(task_files):
            if verbose:
                print(f"\n--- Starting Task {i+1}/{len(task_files)}: {os.path.dirname(task_file).split('/')[-2:]} ---")
            
            # The "problem" for ALFWorld is the path to the task file
            problem_data = {"task_file": task_file}
            
            # Since this is an async method, we await the evaluation of each problem.
            # We pass the shared agent and env instances.
            result_entry = await self.evaluate_problem(agent, problem_data, env=env, verbose=verbose)
            all_results.append(result_entry)
        
        env.close()

        # 6. Aggregate final results
        summary = self.aggregate_results(all_results)
        summary["benchmark"] = "alfworld"
        summary["agent_system"] = agent_system

        if verbose:
            self.print_summary(summary)

        return summary

    async def evaluate_problem(self, agent, problem: Dict[str, Any], env: AlfredThorEnv, verbose: bool = True) -> Dict[str, Any]:
        """
        Runs a single ALFWorld task.
        """
        task_file = problem["task_file"]
        
        obs, reset_info = env.reset(task_file)
        if reset_info and reset_info.get("error"):
            print(f"Error resetting environment for task: {task_file}")
            return {
                "problem_id": os.path.basename(os.path.dirname(task_file)),
                "problem": task_file,
                "expected": "Success",
                "prediction": "Failure",
                "score": 0,
                "is_correct": False,
                "status": "error",
                "summary": {"error_message": obs}
            }

        done = False
        steps = 0
        agent_response = {} # Ensure it's defined
        trajectory = []
        simple_trajectory = []

        # Start the agent's task, initializing its internal state (e.g., message history)
        await agent.start_task(problem)
        
        if verbose:
            print(f"Initial observation: {obs}")
        
        try:
            while not done:
                info_tuple = env.get_info()
                # Safely get game_info, ensuring it's a dict, defaulting to {} if it's None or the tuple is too short.
                game_info = (info_tuple[3] if len(info_tuple) > 3 else None) or {}
                admissible_commands = game_info.get('admissible_commands', [])
                
                if verbose:
                    print(f"\n--- Step {steps+1} ---")
                    print(f"Admissible commands: {admissible_commands}")

                problem_prompt = (
                    f"Your task is to:\n{obs}\n\n"
                    f"You have observed this scene. Now, choose your next action from the list of admissible commands.\n"
                    f"Admissible commands: {admissible_commands}\n\n"
                    f"Your response MUST BE one of the commands from the list above, and nothing else. Do not add any reasoning, explanation, or formatting."
                )
                
                try:
                    if verbose:
                        print("Querying agent for next action...")
                    
                    # Use the new run_step method for multi-turn interaction
                    agent_response = await agent.run_step(prompt=problem_prompt)

                    if verbose:
                        print(f"Agent response received: {agent_response.get('extracted_answer', '')}")
                except Exception as e:
                    import traceback
                    print(f"An error occurred while querying the agent: {e}")
                    print(traceback.format_exc())
                    agent_response = {} # Default to empty response on error

                if not agent_response:
                    print("Warning: Agent returned None response. Defaulting to empty response.")
                    agent_response = {}

                raw_action = agent_response.get("extracted_answer", "").strip()
                action = raw_action

                # Add a simple parser to be more robust against verbose models
                if "**Action:**" in raw_action:
                    try:
                        action = raw_action.split("**Action:**")[-1].strip()
                    except IndexError:
                        pass # Keep the raw_action if parsing fails

                if not action or action not in admissible_commands:
                    if verbose:
                        print(f"Warning: Agent proposed an invalid action '{action}'. It is not in the list of admissible commands. Defaulting to 'look'.")
                    action = "look"

                if verbose:
                    print(f"Step {steps+1}: Action: {action}")

                simple_trajectory.append(f"第{steps+1}步：{action}")

                obs, _, done, info = env.step(action)
                
                trajectory.append({"step": steps + 1, "action": action, "observation": obs, "info": info})

                if verbose:
                    print(f"Observation after action: {obs}")
                    print(f"Info: {info}")
                    
                steps += 1
                
                if steps >= MAX_STEPS:
                    if verbose:
                        print("Max steps reached. Ending task.")
                    done = True
        finally:
            # Ensure the agent's task is always ended to clean up state
            await agent.end_task()

        task_info = env.get_info()[3] or {}
        is_correct = task_info.get('won', False)
        
        result_entry = {
            "problem_id": os.path.basename(os.path.dirname(task_file)),
            "problem": task_info.get('gamefile'),
            "expected": "Success",
            "prediction": "Success" if is_correct else "Failure",
            "score": 1 if is_correct else 0,
            "is_correct": is_correct,
            "status": "completed",
            "llm_usage": agent_response.get("llm_usage", {}),
            "trajectory": trajectory,
            "simple_trajectory": simple_trajectory,
            "summary": {
                "correct": is_correct, 
                "score": 1 if is_correct else 0,
                "goal_condition_success_rate": task_info.get('goal_condition_success_rate', 0.0),
                "steps": steps
            },
        }
        if verbose:
            print(f"--- Task Finished. Success: {is_correct} ---")
        return result_entry

    def aggregate_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates results from all tasks."""
        total = len(all_results)
        if total == 0:
            return {"total": 0, "correct": 0, "accuracy": 0, "errored": 0}

        correct = sum(1 for r in all_results if r.get("is_correct", False))
        errored = sum(1 for r in all_results if r.get("status") == "error")
        accuracy = correct / total if total > 0 else 0
        
        summary = {
            "total": total,
            "correct": correct,
            "errored": errored,
            "accuracy": accuracy,
        }
        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Prints a formatted summary of the evaluation results."""
        print("\n" + "="*50)
        print("ALFWorld Evaluation Summary:")
        print(f"  Agent System: {summary.get('agent_system', 'N/A')}")
        print(f"  Total tasks: {summary.get('total', 0)}")
        print(f"  Successful tasks: {summary.get('correct', 0)}")
        print(f"  Success Rate (Accuracy): {summary.get('accuracy', 0):.3f}")
        print("="*50)
