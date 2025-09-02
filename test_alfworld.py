# test_alfworld.py
import os
import yaml
import glob
from dotenv import load_dotenv
load_dotenv(override=True)

def test_step_1_imports():
    """
    步骤 1: 验证所有主要模块是否可以被成功导入。
    """
    print("--- 步骤 1: 开始基础编译与导入测试 ---")
    try:
        from mas_arena.evaluators.ALFWorld.alfworld_evaluator import AlfWorldEvaluator
        from mas_arena.evaluators.ALFWorld.environment import AlfredThorEnv
        from mas_arena.evaluators.ALFWorld.alfworld_env import thor_env, oracle, game_util
        print("✅ 成功: 所有核心模块都已成功导入。")
        return True
    except ImportError as e:
        print(f"❌ 失败: 导入模块时出错: {e}")
        return False

def test_step_2_env_initialization():
    """
    步骤 2: 验证 ALFWorld 环境是否可以被成功初始化。
    """
    print("\n--- 步骤 2: 开始环境初始化测试 ---")
    env = None
    try:
        # 1. 加载配置文件
        config_file_path = os.path.join(os.path.dirname(__file__), "mas_arena/evaluators/ALFWorld/config.yaml")
        if not os.path.exists(config_file_path):
            # Try a different path if the first one fails
            config_file_path = "mas_arena/evaluators/ALFWorld/config.yaml"
        if not os.path.exists(config_file_path):
            print(f"❌ 失败: 找不到配置文件: {config_file_path}")
            return False
            
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ 成功: 配置文件已加载。")

        # 2. 检查 ALFWORLD_DATA 环境变量，如果未设置则尝试自动检测
        if "ALFWORLD_DATA" not in os.environ:
            print("🟡 警告: ALFWORLD_DATA 环境变量未设置。")
            default_data_path = "mas_arena/evaluators/ALFWorld/data"
            if os.path.exists(default_data_path):
                print(f"   - 发现默认数据路径: '{default_data_path}'")
                os.environ["ALFWORLD_DATA"] = default_data_path
                print(f"   - ✅ 成功: 已自动将 ALFWORLD_DATA 设置为: {default_data_path}")
            else:
                print(f"❌ 失败: 未找到默认数据路径 '{default_data_path}'。")
                print("   请手动设置 ALFWORLD_DATA 环境变量指向您的 ALFWorld 数据目录。")
                print("   例如: export ALFWORLD_DATA=~/alfworld/data")
                return False

        data_path = os.path.expandvars(config['dataset']['data_path'])
        
        # 智能检测 json_2.1.1 子目录
        json_subdir_path = os.path.join(data_path, "json_2.1.1")
        if os.path.isdir(json_subdir_path):
            print(f"   - 发现 'json_2.1.1' 子目录，将自动使用此路径进行搜索。")
            data_path = json_subdir_path

        split = config['dataset']['split']
        task_files_path = os.path.join(data_path, split)
        task_files = glob.glob(os.path.join(task_files_path, "**", "traj_data.json"), recursive=True)

        if not task_files:
            print(f"❌ 失败: 在 '{task_files_path}' 中没有找到任务文件 (traj_data.json)。")
            print("   请确认 ALFWORLD_DATA 环境变量是否指向了正确的路径。")
            return False
        print(f"✅ 成功: 找到了 {len(task_files)} 个任务文件。")

        # 3. 初始化环境
        # 强制指定 AI2THOR 可执行文件路径，避免重新下载
        thor_build_path = os.path.expanduser("~/.ai2thor/releases/thor-201909061227-Linux64")
        if os.path.exists(thor_build_path):
            print(f"   - 发现本地 AI2-THOR 环境: {thor_build_path}")
            os.environ['AI2THOR_LOCAL_BUILD_PATH'] = thor_build_path
        else:
            print("   - 未发现本地 AI2-THOR 环境，将尝试自动下载...")
            
        print("   - 正在初始化 AlfredThorEnv 环境。首次运行时，需要下载 AI2-THOR 可执行文件（约 400MB），请耐心等待...")
        from mas_arena.evaluators.ALFWorld.environment import AlfredThorEnv
        env = AlfredThorEnv(config)
        print("✅ 成功: AlfredThorEnv 环境对象已创建。")

        # 4. 使用第一个任务文件重置环境
        task_file_to_test = task_files[0]
        print(f"   - 正在使用任务文件进行测试: {task_file_to_test}")
        env.reset(task_file_to_test)
        print("✅ 成功: 环境已成功重置并加载任务。")
        
        return True

    except Exception as e:
        print(f"❌ 失败: 环境初始化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if env:
            env.close()
            print("   - 环境资源已清理。")


async def test_step_3_single_task_evaluation():
    """
    步骤 3: 使用 singleagent 运行一个完整的单任务评测流程。
    """
    print("\n--- 步骤 3: 开始单任务完整流程测试 ---")
    try:
        # 1. 定义 agent 配置 (使用 gpt-4o-mini 作为示例)
        # 注意: 请确保您的环境中已设置 OPENAI_API_KEY
        print("   - 正在配置 singleagent...")
        agent_system = "singleagent"
        agent_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "kwargs": {
                    "temperature": 0,
                    "max_tokens": 200
                }
            },
            "max_retries": 2
        }
        print(f"   - ✅ 成功: Agent 配置完成 ({agent_config['llm']['provider']}/{agent_config['llm']['model']})。")
        
        # 2. 导入并运行评测函数，限制只跑一个任务 (limit=1)
        from mas_arena.evaluators.ALFWorld.alfworld_evaluator import AlfWorldEvaluator
        print("   - 正在运行 AlfWorldEvaluator(limit=1)...")
        
        # 需要 asyncio.run 来执行异步函数
        import asyncio
        evaluator = AlfWorldEvaluator()
        summary = await evaluator.run(
            agent_system=agent_system,
            agent_config=agent_config,
            limit=1,
            verbose=True
        )

        # 3. 验证结果
        if summary:
            print("   - ✅ 成功: 评测函数已成功执行并返回结果。")
            print(f"   - 任务结果: {'成功' if summary.get('correct') == 1 else '失败'}")
            print(f"   - 最终摘要: {summary}")
            return True
        else:
            print("   - ❌ 失败: 评测函数未返回有效的结果。")
            return False

    except Exception as e:
        print(f"❌ 失败: 单任务评测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- 在此运行测试 ---
if __name__ == "__main__":
    import asyncio

    if test_step_1_imports():
        print("\n基础编译测试通过。")
        if test_step_2_env_initialization():
            print("\n环境初始化测试通过。")
            # 使用 asyncio.run 来执行异步的测试步骤
            if asyncio.run(test_step_3_single_task_evaluation()):
                print("\n✅ 所有测试步骤均已成功通过！ALFWorld 评测流程已完全跑通。")