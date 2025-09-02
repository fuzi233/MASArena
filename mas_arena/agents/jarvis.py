import os
import time
import re
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry


class Task:
    """Task execution representation."""

    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: str):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool


@dataclass
class Agent:
    """Represents an LLM agent"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        self.chat_history = []
        self.llm = ChatOpenAI(
            model=self.model_name
        )

    async def generate_response(self, context: str) -> Dict[str, Any]:
        """Generate agent response asynchronously"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        try:
            # Use standard output in async mode
            response = await self.llm.ainvoke(messages)
            # Save response name for source identification
            response.name = self.name
            
            # Add to chat history
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": response.content
            })
            
            # Ensure we return the original response object to preserve usage_metadata
            return {
                "message": response,  # Contains the complete AIMessage object with metadata
                "content": response.content
            }
                
        except Exception as e:
            print(f"Response generation failed: {str(e)}")
            
            # Create simple response
            error_content = f"Generation failed: {str(e)}"
            
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": error_content
            })
            
            # Create a minimal AIMessage object
            error_message = AIMessage(content=error_content)
            error_message.name = self.name
            
            return {
                "message": error_message,
                "content": error_content
            }


class JARVIS(AgentSystem):
    """JARVIS multi-agent system"""

    def __init__(self, name: str = "jarvis", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Get model name
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Initialize agents
        self.planner_agent = None
        self.executor_agent = None
        self.response_generator_agent = None
        
        # Initialize task list
        self.tasks = []
        self.task_results = {}

    def _create_agents(self):
        """Create the system's required agents"""
        # Task planning agent
        self.planner_agent = Agent(
            agent_id="planner",
            name="Task Planner",
            model_name=self.model_name,
            system_prompt="""You are an expert task planner specialized in problem decomposition.
Your job is to analyze problems and create a detailed task plan for solving them.

For each task, provide:
1. Task ID (starting from 1)
2. Task description (clearly stating what needs to be accomplished)
3. Dependency tasks (list of IDs of tasks that must be completed before this task can start)

Ensure dependencies between tasks are clear, and the task sequence effectively solves the problem.
Keep tasks linear and avoid complex parallel processing.

Be especially careful with:
- Mathematical problems: Break them down into clear computational steps
- Reasoning problems: Create steps for analysis, inference, and conclusion
- Complex problems: Decompose into smaller, manageable sub-problems

Always ensure the final task synthesizes all previous task results."""
        )
        
        # Task execution agent
        self.executor_agent = Agent(
            agent_id="executor",
            name="Task Executor",
            model_name=self.model_name,
            system_prompt="""You are an expert task executor specialized in precise problem-solving.
Your job is to execute specified tasks and provide detailed execution results.

When receiving a task description and related parameters, you must:
1. Understand the task objective
2. Perform necessary calculations or reasoning with absolute precision
3. Show your detailed step-by-step work
4. Provide clear execution results

Ensure your results are accurate, your reasoning process is clear, and your output can be used for subsequent tasks."""
        )
        
        # Response generation agent
        self.response_generator_agent = Agent(
            agent_id="generator",
            name="Response Generator",
            model_name=self.model_name,
            system_prompt=f"""You are an expert response generator specialized in synthesis and precision.
Your job is to generate a final comprehensive answer based on all previous task results.

When receiving a problem description and all completed task results, you must:
- Synthesize and analyze all task results comprehensively
- Provide a clear, accurate final answer, and step-by-step solution
{self.format_prompt}
"""
        )

    def _parse_tasks_from_text(self, text: str) -> List[Task]:
        """Parse task list from text"""
        tasks = []
        try:
            # Try multiple regex patterns to catch different task formats
            task_patterns = [
                r"Task\s*(\d+)[:\s]+\s*(.*?)(?=Task\s*\d+[:\s]+|$)",
                r"(\d+)\.\s*Task[:\s]+\s*(.*?)(?=\d+\.\s*Task[:\s]+|$)",
                r"(\d+)\.\s*(.*?)(?=\d+\.\s*|$)"
            ]
            
            tasks_found = []
            for pattern in task_patterns:
                tasks_found = re.findall(pattern, text, re.DOTALL)
                if tasks_found:
                    break
            
            for task_id_str, task_desc in tasks_found:
                task_id = int(task_id_str)
                
                # Try to extract dependencies from description
                dep_patterns = [
                    r"Depends on[:\s]+\s*\[(.*?)\]",
                    r"Dependencies[:\s]+\s*\[(.*?)\]",
                    r"Dependency[:\s]+\s*\[(.*?)\]"
                ]
                
                deps = []
                for pattern in dep_patterns:
                    dep_match = re.search(pattern, task_desc)
                    if dep_match:
                        deps_str = dep_match.group(1)
                        deps = [int(d.strip()) for d in deps_str.split(",") if d.strip().isdigit()]
                        break
                
                # Create task
                task = Task(
                    task=task_desc.strip(),
                    id=task_id,
                    dep=deps,
                    args={"description": task_desc.strip()},
                    tool="reasoning"
                )
                tasks.append(task)
            
            # If no tasks found, create a default task
            if not tasks:
                tasks.append(Task(
                    task="Solve the problem",
                    id=1,
                    dep=[],
                    args={"description": "Analyze and solve the given problem"},
                    tool="reasoning"
                ))
                
        except Exception as e:
            print(f"Error parsing tasks: {str(e)}")
            tasks.append(Task(
                task="Solve the problem",
                id=1,
                dep=[],
                args={"description": "Analyze and solve the given problem"},
                tool="reasoning"
            ))
        
        return tasks

    async def _plan_tasks(self, problem: str) -> List[Task]:
        """Plan task list asynchronously"""
        if not self.planner_agent:
            self._create_agents()
            
        prompt = f"""
        Please analyze the following problem and create a task plan for solving it:

        Problem: {problem}
        
        Provide a list of tasks, each including:
        1. Task ID
        2. Task description
        3. List of dependency task IDs (empty list if no dependencies)
        
        Format example:
        Task 1: Analyze the problem
        Dependencies: []
        
        Task 2: Solve sub-problem A
        Dependencies: [1]
        
        Task 3: Solve sub-problem B
        Dependencies: [1]
        
        Task 4: Merge results
        Dependencies: [2, 3]
        
        Ensure tasks are linear and don't require complex parallel processing.
        For mathematical problems, break down each computational step clearly.
        """
        
        response = await self.planner_agent.generate_response(prompt)
        content = response.get("content", "")
        
        return self._parse_tasks_from_text(content)

    async def _execute_task(self, task: Task, problem: str) -> Dict[str, Any]:
        """Execute a single task asynchronously"""
        if not self.executor_agent:
            self._create_agents()
            
        # Prepare results from dependent tasks
        dependency_results = {}
        for dep_id in task.dep:
            if dep_id in self.task_results:
                dependency_results[f"Task {dep_id} Result"] = self.task_results[dep_id].get("content", "")
        
        dependency_text = "\n\n".join([f"{name}:\n{result}" for name, result in dependency_results.items()])
        
        prompt = f"""
        Original Problem: {problem}
        
        Current Task:
        ID: {task.id}
        Description: {task.task}
        
        Results from Dependency Tasks:
        {dependency_text if dependency_text else "No dependency tasks"}
        
        Please execute this task with precision and provide detailed step-by-step work and results.
        For mathematical calculations, show all steps and verify your final answer.
        For reasoning tasks, explain your logical process thoroughly.
        """
        
        response = await self.executor_agent.generate_response(prompt)
        
        execution_result = {
            "task_id": task.id,
            "content": response.get("content", ""),
            "message": response.get("message", None)  # Preserve complete message object with metadata
        }
        
        return execution_result

    async def _generate_final_response(self, problem: str, task_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final response asynchronously"""
        if not self.response_generator_agent:
            self._create_agents()
            
        # Format all task results
        formatted_results = "\n\n".join([
            f"Task {task_id} Result:\n{result.get('content', '')}"
            for task_id, result in task_results.items()
        ])
        
        prompt = f"""
        Original Problem: {problem}
        
        Task Execution Results:
        {formatted_results}
        
        Based on all the information above, please generate a comprehensive final answer.
        
        Your response must:
        - Verify all calculations from the previous tasks
        - {self.format_prompt}
        
        Make sure your answer is concise, logically clear, and directly answers the original problem.
        Double-check all mathematical operations for accuracy.
        """
        
        response = await self.response_generator_agent.generate_response(prompt)
        content = response.get("content", "")
        
        return {
            "final_answer": content,
            "message": response.get("message", None)  # Preserve complete message object with metadata
        }

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the agent system to process a problem asynchronously"""
        print("JARVIS: Starting problem processing...")
        start_time = time.time()
        
        # Extract problem text
        problem_text = problem.get("problem", "")
        print(f"Problem: {problem_text[:100]}...")
        
        # Initialize agents
        self._create_agents()
        print("Agents created successfully")
        
        # Step 1: Task planning
        print("Starting task planning...")
        self.tasks = await self._plan_tasks(problem_text)
        print(f"Task planning complete, {len(self.tasks)} tasks planned")
        
        # Collect all messages - containing complete AIMessage objects
        all_messages = []
        planner_message = None
        if self.planner_agent.chat_history:
            planner_content = self.planner_agent.chat_history[-1]["ai"]
            # If it's an AIMessage object, preserve it; otherwise create one
            if hasattr(planner_content, "content"):
                planner_message = planner_content
            else:
                # Create an AIMessage object
                msg = AIMessage(content=str(planner_content))
                msg.name = "Task Planner"
                planner_message = msg
            
            all_messages.append(planner_message)
            print(f"Added planner agent response, length: {len(str(planner_message.content))}")
        else:
            print("Warning: Planner agent didn't generate history")
            msg = AIMessage(content="Task planning didn't generate valid output")
            msg.name = "Task Planner"
            all_messages.append(msg)
        
        # Step 2: Execute tasks with dependency-based parallelization
        print("Starting task execution...")
        self.task_results = {}
        
        # Track which tasks are completed
        completed_tasks = set()
        pending_tasks = {task.id: task for task in self.tasks}
        
        while pending_tasks:
            # Find tasks that can be executed in parallel (all dependencies satisfied)
            ready_tasks = []
            for task_id, task in list(pending_tasks.items()):
                dependencies_satisfied = all(dep_id in completed_tasks for dep_id in task.dep)
                if dependencies_satisfied:
                    ready_tasks.append(task)
                    del pending_tasks[task_id]
            
            if not ready_tasks:
                # If no tasks are ready but there are pending tasks, there might be a dependency cycle
                print("Warning: Possible dependency cycle detected. Breaking cycle...")
                # Force-execute the first pending task
                if pending_tasks:
                    first_pending = next(iter(pending_tasks.values()))
                    ready_tasks.append(first_pending)
                    del pending_tasks[first_pending.id]
                else:
                    # All tasks are done
                    break
            
            print(f"Executing {len(ready_tasks)} tasks in parallel...")
            
            # Execute all ready tasks in parallel
            tasks_execution = [self._execute_task(task, problem_text) for task in ready_tasks]
            results = await asyncio.gather(*tasks_execution)
            
            # Process results
            for i, result in enumerate(results):
                task = ready_tasks[i]
                task_id = task.id
                
                self.task_results[task_id] = result
                completed_tasks.add(task_id)
                print(f"Task {task_id} completed")
                
                # Add to message list - preserve complete AIMessage object
                if "message" in result and result["message"] is not None:
                    all_messages.append(result["message"])
                    print(f"Added execution message to results, length: {len(str(result['message'].content))}")
                else:
                    # Create an AIMessage object
                    content = result.get("content", f"Task {task_id} execution result unavailable")
                    msg = AIMessage(content=content)
                    msg.name = f"Task Executor {task_id}"
                    all_messages.append(msg)
                    print(f"Added execution message to results, length: {len(content)}")
        
        # Step 3: Generate final response
        print("Generating final response...")
        final_response = await self._generate_final_response(problem_text, self.task_results)
        
        # Add final response to message list - preserve complete AIMessage object
        if "message" in final_response and final_response["message"] is not None:
            all_messages.append(final_response["message"])
            print(f"Added final response to results, length: {len(str(final_response['message'].content))}")
        else:
            # Create an AIMessage object
            content = final_response.get("final_answer", "Final response generation failed")
            msg = AIMessage(content=content)
            msg.name = "Response Generator"
            all_messages.append(msg)
            print(f"Added final response to results, length: {len(content)}")
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Processing complete, took {duration_ms:.2f}ms")
        
        # Build result
        result = {
            "problem_id": problem.get("id", ""),
            "problem": problem_text,
            "messages": all_messages,  # Preserve complete AIMessage object list
            "final_answer": final_response.get("final_answer", ""),
            "execution_time_ms": duration_ms,
        }
        
        # Record agent responses - pass complete AIMessage object list to preserve usage_metadata
        try:
            print("Recording agent responses...")
            self._record_agent_responses(result["problem_id"], all_messages)
        except Exception as e:
            print(f"Error recording responses: {str(e)}")
        
        print("JARVIS processing complete")
        return result


# Register agent system
AgentSystemRegistry.register("jarvis", JARVIS)

if __name__ == "__main__":
    # Test the JARVIS agent
    problem = {
        "id": "1",
        "problem": "What is the sum of the first 100 natural numbers?"
    }
    
    jarvis = JARVIS()
    result = jarvis.run_agent(problem)
    print(result)

