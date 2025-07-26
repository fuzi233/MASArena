import time
import json
import os
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, TypedDict, Any, List, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

load_dotenv()

@dataclass
class DebateAgent:
    """表示一个辩论参与者"""
    agent_id: str
    name: str
    model_name: str
    temperature: float
    memory_lst: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.memory_lst is None:
            self.memory_lst = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            request_timeout=60,
            max_retries=2
        )

    def set_meta_prompt(self, meta_prompt: str):
        """设置元提示"""
        self.memory_lst.append({"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        """添加新事件到记忆中"""
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        """添加生成的回复到记忆中"""
        self.memory_lst.append({"role": "assistant", "content": memory})

    def ask(self):
        """查询并获取回答"""
        from langchain_core.messages import AIMessage
        
        messages = []
        for msg in self.memory_lst:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        response = self.llm.invoke(messages)
        response.name = self.name
        response.id = self.agent_id
        return response

@dataclass
class ResultExtractor:
    """结果提取器"""
    model_name: str
    format_prompt: str = ""
    
    def __post_init__(self):
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,
            max_retries=2
        )
        self.name = "result_extractor"
    
    def extract(self, agent_histories: List[List[Dict]], problem_text: str):
        """从agent历史中提取最终答案"""
        # 构建提取提示
        extract_prompt = f"""Based on the debate history, please extract the final answer to the following problem:

Problem: {problem_text}

{self.format_prompt}

Please provide only the final answer."""
        
        messages = [HumanMessage(content=extract_prompt)]
        response = self.llm.invoke(messages)
        return {"message": response}

class MADAgent(AgentSystem):
    """Multi-Agent Debate系统"""
    
    def __init__(self, name: str = "mad", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.num_players = self.config.get("num_players", 3)
        self.max_round = self.config.get("max_round", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0)
        
        # 辩论配置
        self.debate_config = {
            "debate_topic": "",
            "base_answer": "",
            "debate_answer": "",
            "player_meta_prompt": "You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.\nThe debate topic is stated as follows:\n##debate_topic##",
            "moderator_meta_prompt": "You are a moderator. There will be two debaters involved in a debate. They will present their answers and discuss their perspectives on the following topic: \"##debate_topic##\"\nAt the end of each round, you will evaluate answers and decide which is correct.",
            "affirmative_prompt": "##debate_topic##",
            "negative_prompt": "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons.",
            "moderator_prompt": "Now the ##round## round of debate for both sides has ended.\n\nAffirmative side arguing:\n##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nYou, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude. If not, the debate will continue to the next round. Now please output your answer in json format, with the format as follows: {\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content.",
            "judge_prompt_last1": "Affirmative side arguing: ##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nNow, what answer candidates do we have? Present them without reasons.",
            "judge_prompt_last2": "Therefore, ##debate_topic##\nPlease summarize your reasons and give the final answer that you think is correct. Now please output your answer in json format, with the format as follows: {\"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content.",
            "debate_prompt": "##oppo_ans##\n\nDo you agree with my perspective? Please provide your reasons and answer."
        }
        
        # 初始化组件
        agent_components = self._create_agents()
        self.players = [w for w in agent_components["workers"] if isinstance(w, DebateAgent)]
        extractors = [w for w in agent_components["workers"] if isinstance(w, ResultExtractor)]
        if extractors:
            self.extractor = extractors[0]
        else:
            self.extractor = ResultExtractor(self.model_name, self.format_prompt)

    def _create_agents(self) -> Dict[str, List]:
        """创建辩论参与者和结果提取器"""
        name_list = ["Affirmative side", "Negative side", "Moderator"]
        
        players = []
        for i, name in enumerate(name_list):
            agent = DebateAgent(
                agent_id=f"agent_{i+1}",
                name=name,
                model_name=self.model_name,
                temperature=self.temperature
            )
            players.append(agent)
        
        # 创建结果提取器
        extractor = ResultExtractor(self.model_name, self.format_prompt)
        
        return {
            "workers": players + [extractor]
        }

    def init_prompt(self, debate_topic: str):
        """初始化并替换提示模板中的占位符"""
        config = self.debate_config.copy()
        for key in config:
            if isinstance(config[key], str):
                config[key] = config[key].replace("##debate_topic##", debate_topic)
        return config

    def round_dct(self, num: int) -> str:
        """将数字转换为序数词"""
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return dct.get(num, str(num))

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """运行辩论过程"""
        problem_text = problem["problem"]
        
        # 存储所有LLM响应
        all_messages = []
        
        # 使用format_prompt作为辩论主题
        debate_topic = f"{problem_text}\n\n{self.format_prompt}" if self.format_prompt else problem_text
        
        # 初始化提示
        config = self.init_prompt(debate_topic)
        
        # 获取参与者
        affirmative = self.players[0]
        negative = self.players[1] 
        moderator = self.players[2]
        
        # 设置元提示
        affirmative.set_meta_prompt(config['player_meta_prompt'])
        negative.set_meta_prompt(config['player_meta_prompt'])
        moderator.set_meta_prompt(config['moderator_meta_prompt'])
        
        # 第一轮辩论
        affirmative.add_event(config['affirmative_prompt'])
        aff_response = affirmative.ask()
        affirmative.add_memory(aff_response.content)
        all_messages.append(aff_response)
        aff_ans = aff_response.content
        
        negative.add_event(config['negative_prompt'].replace('##aff_ans##', aff_ans))
        neg_response = negative.ask()
        negative.add_memory(neg_response.content)
        all_messages.append(neg_response)
        neg_ans = neg_response.content
        
        moderator.add_event(config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', 'first'))
        mod_response = moderator.ask()
        moderator.add_memory(mod_response.content)
        all_messages.append(mod_response)
        
        try:
            mod_ans = json.loads(mod_response.content)
        except:
            mod_ans = {"debate_answer": "", "Whether there is a preference": "No"}
        
        # 多轮辩论
        for round_num in range(2, self.max_round + 1):
            if mod_ans.get("debate_answer", "") != "":
                break
                
            affirmative.add_event(config['debate_prompt'].replace('##oppo_ans##', neg_ans))
            aff_response = affirmative.ask()
            affirmative.add_memory(aff_response.content)
            all_messages.append(aff_response)
            aff_ans = aff_response.content
            
            negative.add_event(config['debate_prompt'].replace('##oppo_ans##', aff_ans))
            neg_response = negative.ask()
            negative.add_memory(neg_response.content)
            all_messages.append(neg_response)
            neg_ans = neg_response.content
            
            moderator.add_event(config['moderator_prompt'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans).replace('##round##', self.round_dct(round_num)))
            mod_response = moderator.ask()
            moderator.add_memory(mod_response.content)
            all_messages.append(mod_response)
            
            try:
                mod_ans = json.loads(mod_response.content)
            except:
                mod_ans = {"debate_answer": "", "Whether there is a preference": "No"}
        
        # 如果仍未达成一致，使用裁判
        final_answer = mod_ans.get("debate_answer", "")
        if not final_answer:
            judge = DebateAgent(
                agent_id="judge",
                name="Judge",
                model_name=self.model_name,
                temperature=self.temperature
            )
            judge.set_meta_prompt(config['moderator_meta_prompt'])
            
            # 获取最终答案候选
            judge.add_event(config['judge_prompt_last1'].replace('##aff_ans##', aff_ans).replace('##neg_ans##', neg_ans))
            judge_response1 = judge.ask()
            judge.add_memory(judge_response1.content)
            all_messages.append(judge_response1)
            
            # 选择最终答案
            judge.add_event(config['judge_prompt_last2'])
            judge_response2 = judge.ask()
            judge.add_memory(judge_response2.content)
            all_messages.append(judge_response2)
            
            try:
                judge_ans = json.loads(judge_response2.content)
                final_answer = judge_ans.get("debate_answer", judge_response2.content)
            except:
                final_answer = judge_response2.content
        
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

# 注册agent系统
AgentSystemRegistry.register(
    "mad",
    MADAgent,
    num_players=3,
    max_round=3,
    temperature=0
) 