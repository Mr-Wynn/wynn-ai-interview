"""
面试系统 Graph 定义
"""

import json
import logging
from typing import List, TypedDict, Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from app.core import interview_rag, llms
from app.core.memory import get_async_sqlite_saver
from app.core import interview_analysis
from app.utils.response_util import parse_response_dict, parse_response_str

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

class InterviewState(TypedDict):
    """
    面试状态定义 - 统一的状态结构
    """
    # 基础配置
    session_id: str
    resume_path: str  # 原始简历路径
    job_description: Optional[str]  # 面试岗位（如 LLM 应用开发工程师）
    company_info: Optional[str]  # 公司信息
    interview_focus: Optional[str]  # 面试侧重：project/general/balanced
    api_config: dict  # 大模型配置

    # 初始化节点状态变更
    resume_projects: Optional[List[dict]]  # 自动解析的项目列表：[{"name":"XX项目","tech_stack":["Python","LangChain"],"desc":"职责"}]
    project_tech_points: List[str]  # 项目相关技术点（从项目中提取，如 ["RAG 检索优化", "智能体协作"]）
    general_tech_points: List[str]  # 岗位通用考点清单（如 ["RAG 分块策略", "ReAct 模式", "LangGraph 节点设计"]）
    current_exam_point_type: Optional[str]  # 当前题型类型：general（通用题）/ project（项目题）
    resume_text: Optional[str]  # 解析的简历内容
    interview_status: Optional[str]  # 进行中/结束

    # 检索节点状态变更
    rag_results: List[str]  # RAG 检索的常规题+技术点素材
    current_exam_point: Optional[str] # 当前考点

    # 问题生成节点状态变更
    current_question: Optional[str]  # 当前面试题

    # 执行面试节点状态变更
    need_user_answer: bool  # 是否需要用户回答
    current_answer: Optional[str]  # 当前用户的回答
    answer_quality: Optional[str]  # 回答质量
    current_point_history: List[dict]  # 当前考点历史对话：[{"type":"题型","q":"问题","a":"回答"}]
    history: List[dict]  # 总历史对话：[{"type":"题型","q":"问题","a":"回答"}]
    
    # 反应决策节点状态变更
    completed_points: List[str]  # 已完成的考点/项目技术点（去重）
    react_decision: Optional[str]  # ReAct 决策指令
    question_count: int  # 已完成的问题数
    follow_up_reason: Optional[str]  # 追问原因

    # 总结分析
    weak_points: Optional[str]  # 薄弱点（通用考点/项目技术点）
    interview_result: Optional[str]
    score_details: Optional[str]
    # 统计信息
    max_questions: int
    # 追问控制
    follow_up_count: int  # 当前主线问题的追问次数
    max_follow_ups: int

# ============================================================================
# 图构建
# ============================================================================

async def init_interview(state: InterviewState):
    """
    初始化面试：
        1. 解析简历内容（支持PDF）
        2. 使用大模型分析简历、岗位、面试侧重，生成考点计划
        3. 初始化所有 State 字段，设置面试状态为「进行中」
    """

    logger.info(f"开始初始化面试会话: {state.get('session_id', 'unknown')}")
    # ============================================================================
    # 1. 解析简历内容（支持PDF文本和直接文本）
    # ============================================================================
    # ["resume_context", "job_description", "company_info", "interview_focus", "general_topics_str"]
    resume_context = await interview_analysis.extract_resume_text(state.get("resume_path", ""))
    analysis_context = {
        "resume_context": resume_context,
        "job_description": state.get("job_description", ""),
        "company_info": state.get("company_info", ""),
        "interview_focus": state.get("interview_focus", ""),
        "general_topics_str": state.get("general_topics_str", "无"),
    }
    analysis_result = await interview_analysis.analyze_resume_with_llm(analysis_context, state.get("api_config"))

    # 构建初始状态更新
    initial_state = {
        "resume_projects": analysis_result.get("resume_projects", []),
        "project_tech_points": analysis_result.get("project_tech_points"),
        "general_tech_points": analysis_result.get("general_tech_points"),
        "current_exam_point_type": analysis_result.get("first_topic_type"),
        "interview_status": "running",
        "resume_text": resume_context
    }
    return initial_state

async def rag_retrieve(state: InterviewState):
    """
    检索知识库：
        1. 根据优先级和题型选择考点
        2. 从知识库中召回常规面试题 + 技术点定义
    """
    logger.info(f"【检索节点】开始执行RAG检索，会话ID: {state.get('session_id', 'unknown')}")
    current_exam_point_type = state["current_exam_point_type"]
    project_tech_points = state["project_tech_points"]
    general_tech_points = state["general_tech_points"]
    completed_points = state["completed_points"]

    uncompleted_project = [tp for tp in project_tech_points if tp not in completed_points]
    uncompleted_general = [tp for tp in general_tech_points if tp not in completed_points]

    current_exam_point = None
    if current_exam_point_type == "project" and uncompleted_project:
        current_exam_point = uncompleted_project[0]
    elif current_exam_point_type == "general" and uncompleted_general:
        current_exam_point = uncompleted_general[0]
    else:
        current_exam_point = uncompleted_general[0] if uncompleted_general else uncompleted_project[0]

    rag_results = await interview_rag.rag_retrieve_core(single_topic=current_exam_point, state=state)

    update_state = {
        "current_exam_point": current_exam_point,
        "rag_results": rag_results
    }
    return update_state
async def generate_question(state: InterviewState):
    """
        面试问题生成核心节点 - 双分支核心逻辑
        分支1：来自【检索节点rag_retrieve】→ 生成全新面试题（新考点问题）
            入参依赖：current_exam_point、rag_results、current_exam_point_type、job_description、resume_text
        分支2：来自【决策节点react_decision】→ 生成知识点追问题（同考点递进）
            入参依赖：current_point_history、follow_up_reason、current_exam_point
        【重要原则】所有检索素材(rag_results)仅作参考，绝不照搬原文，全新生成贴合场景的问题
    """
    logger.info(f"进入问题生成节点 | 会话ID: {state.get('session_id')}")

    current_exam_point = state.get("current_exam_point")
    api_config = state.get("api_config", {})
    llm = llms.get_llm_for_request(api_config, channel="smart")
    # ===================== 核心分支1：判断入边来源 - 从【检索节点】过来 → 生成【全新面试题】 =====================
    if "followup" != state.get("react_decision", None):
        logger.info(
            f"检测到检索节点入边，生成【新考点问题】| 当前考点: {current_exam_point} | 题型: {state.get('current_exam_point_type')}")

        # 新问题专用Prompt - 强制要求【不照搬素材、贴合简历+岗位、考点精准】
        prompt_param = {
            "current_exam_point": state.get("current_point_history", ""),
            "resume_text": state.get("resume_text", ""),
            "job_description": state.get("job_description", ""),
            "rag_results": '\n\t- '.join(state.get("rag_results", [])),
            "current_exam_point_type": state.get("current_exam_point_type", "")
        }
        prompt = PromptTemplate.from_file("resources/generate_question_step1.txt", "utf-8")
        final_prompt = prompt.format(**prompt_param)
        # 调用大模型生成问题
        generated_question = await llm.ainvoke(final_prompt)

    # ===================== 核心分支2：判断入边来源 - 从【决策节点】过来 → 生成【追问面试题】 =====================
    # 判定依据：react_decision执行后必有follow_up_reason+current_point_history，是同考点追问
    else:
        logger.info(
            f"检测到决策节点入边，生成【考点追问问题】| 当前考点: {current_exam_point} | 追问原因: {state.get('follow_up_reason')}")

        # 追问专用Prompt - 强制要求【基于历史对话、递进深挖、不重复原问题】
        prompt_param = {
            "current_exam_point" : state.get("current_exam_point", ""),
            "follow_up_reason": state.get("follow_up_reason", ""),
            "current_point_history": '\n\t- '.join(json.dumps(state.get("follow_up_reason", '{}'))),
            "resume_text": state.get("resume_text", ""),
            "job_description": state.get("job_description", ""),
            "rag_results": '\n\t- '.join(json.dumps(state.get("rag_results", '[]'))),
            "current_exam_point_type": state.get("current_exam_point_type", "")
        }
        prompt = PromptTemplate.from_file("resources/generate_question_step2.txt", "utf-8")
        final_prompt = prompt.format(**prompt_param)
        # 调用大模型生成追问问题
        generated_question = await llm.ainvoke(final_prompt)

    generated_question = parse_response_str(generated_question.content)
    # ===================== 状态更新 - 核心数据落库 =====================
    # 过滤空值，保证生成的问题有效
    updated_state = {
        "need_user_answer" : True,
        "current_question": generated_question.strip() if generated_question else f"请谈谈你对【{current_exam_point}】的理解和相关实践？"
    }
    logger.info(f"问题生成完成 | 当前问题: {updated_state['current_question']}")
    return updated_state


async def execute_interview(state: InterviewState):
    """
    执行面试节点核心逻辑：
    1. 未回答：保持等待状态，提示用户回答当前问题
    2. 已回答：记录问答历史，更新状态并准备进入反应决策节点
    """
    # 1. 日志记录与状态初始化
    session_id = state.get("session_id", "unknown")
    logger.info(f"执行面试节点 - 会话ID: {session_id}")

    api_config = state.get("api_config", {})
    llm = llms.get_llm_for_request(api_config, channel="smart")
    # 2. 提取核心判断变量
    need_user_answer = state.get("need_user_answer", True)
    current_answer = state.get("current_answer", "")
    answer_quality = state.get("answer_quality", "")
    current_point_history = state.get("current_point_history", [])
    history = state.get("history", [])
    # 3. 核心逻辑：判断用户是否已回答
    if not need_user_answer:
        # 3.1 用户已回答：记录问答历史 + 更新状态
        logger.info(f"用户已回答问题 - 会话ID: {session_id}")
        # 调用评估回答质量
        prompt_param = {
            "job_description": state.get("job_description", ""),
            "current_question": state.get("current_question", ""),
            "current_answer": state.get("current_answer", ""),
            "current_point_history": "\n".join(json.dumps(current_point_history)) if current_point_history else "无"
        }
        prompt = PromptTemplate.from_file("resources/interview_assessment.txt", "utf-8")
        final_prompt = prompt.format(**prompt_param)
        answer_quality = await llm.ainvoke(final_prompt)
        # 构建当前问答记录
        qa_record = {
            "current_exam_point_type": state.get("current_exam_point_type", ""),
            "current_exam_point": state.get("current_exam_point", ""),
            "question": state.get("current_question", ""),
            "current_answer": state.get("current_answer", ""),
            "answer_quality": state.get("answer_quality", ""),
            "follow_up_count": state.get("follow_up_count", 0),
        }
        # 更新当前考点历史和总历史
        current_point_history.append(qa_record)
        history.append(qa_record)

    update_state = {
        "current_answer": current_answer,
        "answer_quality": parse_response_str(answer_quality.content),
        "current_point_history": current_point_history,
        "history": history
    }
    return update_state

async def react_decision(state: InterviewState):
    """
        反应决策节点核心逻辑：
            1. 收集面试上下文信息，构建大模型决策提示
            2. 调用大模型评估回答质量并决策下一步动作
            【追问细节】	followup：面试题生成节点
            【切换考点】	switch_topic：rag节点
            【切换题型】	switch_type：rag节点
            【结束本轮】	end：面试评分节点
            3. 更新面试状态（问答记录、追问次数、已完成考点等）
    """
    logger.info(f"执行反应决策节点 - 会话ID: {state.get('session_id', 'unknown')}")
    history = state.get("history", [])
    # ===================== 步骤1：收集决策所需的上下文信息 =====================
    decision_context = {
        "job_description": state.get("job_description", ""),  # 岗位信息
        "interview_focus": state.get("interview_focus", ""),  # 面试侧重
        "current_exam_point_type": state.get("current_exam_point_type", "general"),  # 当前题型
        "current_exam_point": state.get("current_exam_point", ""),  # 当前考点
        "current_question": state.get("current_question", ""),  # 当前问题
        "current_answer": state.get("current_answer", ""),  # 用户回答
        "follow_up_count": state.get("follow_up_count", 0),  # 已追问次数
        "history": "\n".join(json.dumps(history)) if history else "第一道题，无历史对话"
    }

    # ===================== 步骤2：调用大模型获取决策结果 =====================
    prompt = PromptTemplate.from_file("resources/react_decision.txt", "utf-8")
    final_prompt = prompt.format(**decision_context)
    llm = llms.get_llm_for_request(state.get("api_config"), channel="smart")
    react_decision_resp = await llm.ainvoke(final_prompt)
    react_decision_map = parse_response_dict(react_decision_resp.content)
    decision = react_decision_map.get("decision", "end")
    update_state = {}
    # ===================== 步骤3：更新面试状态 =====================
    if decision == "followup":
        # 追问：次数+1
        update_state["follow_up_count"] = state.get("follow_up_count", 0) + 1
        update_state["follow_up_reason"] = react_decision_map.get("follow_up_reason", "")
    else:
        completed_points = state.get("completed_points", [])
        completed_points.append(state.get("current_exam_point", ""))
        update_state["follow_up_count"] = 0
        update_state["follow_up_reason"] = ""
        update_state["current_point_history"] = []
    if decision == "switch_type":
        update_state["current_exam_point_type"] = "project" if state.get( "current_exam_point_type", "") == "general" else "general"

    update_state["react_decision"] = decision
    follow_up_reason = react_decision_map.get("follow_up_reason", "")
    update_state["follow_up_reason"] = follow_up_reason
    update_state["current_answer"] = ""
    # ===================== 步骤4：日志记录并返回更新后的状态 =====================
    logger.info(
        f"反应决策完成 - 会话ID: {state['session_id']}, "
        f" 决策: {decision}, "
        f"追问原因: {follow_up_reason[:50]}..." if follow_up_reason else ""
    )
    return update_state

async def score_evaluation(state: InterviewState):
    """
        评分评估节点核心逻辑：
        1. 从文件加载评分Prompt模板
        2. 构造评估上下文（面试历史/考点/岗位信息）
        3. 调用LLM完成分题型评分+薄弱点识别
        4. 更新状态中的薄弱点和面试状态
    """
    logger.info(f"开始评分评估 | 会话ID: {state['session_id']}")
    eval_context = {
        "job_description": state["job_description"],
        "interview_focus": state["interview_focus"],
        "interview_history": json.dumps(state["history"], ensure_ascii=False, indent=2),
        "completed_points": state["completed_points"]
    }
    # 调用LLM生成复盘报告
    score_prompt = PromptTemplate.from_file("resources/score_evaluation.txt", "utf-8")
    final_prompt = score_prompt.format(**eval_context)
    # 调用大模型评分
    llm = llms.get_llm_for_request(state.get("api_config"), channel="smart")
    score_evaluate_response = await llm.ainvoke(final_prompt)
    score_evaluate = parse_response_dict(score_evaluate_response.content)

    weak_points = score_evaluate.get("weak_points", "")
    score_details = score_evaluate.get("score_details", "")

    updated_state = {
        "weak_points": weak_points,
        "score_details":  score_details
    }
    return updated_state

async def summary_feedback(state: InterviewState):
    """
        总结反馈节点核心逻辑：
        1. 从文件加载总结Prompt模板
        2. 构造复盘上下文（评分结果/薄弱点/面试历史）
        3. 调用LLM生成结构化复盘报告
        4. 保存报告到状态并标记面试完成
    """
    logger.info(f"开始生成总结反馈 | 会话ID: {state['session_id']}")

    # 1. 加载总结Prompt模板
    summary_prompt = PromptTemplate.from_file("resources/summary_feedback.txt", "utf-8")
    # 2. 构造LLM输入上下文
    summary_context = {
        "job_description": state["job_description"],
        "company_info": state["company_info"],
        "resume_text": state["resume_text"],
        "interview_history": json.dumps(state["history"], ensure_ascii=False, indent=2),
        "weak_points": json.dumps(state["weak_points"], ensure_ascii=False),
        "interview_focus": state["interview_focus"],
        "score_details": state["interview_focus"]
    }
    # 3. 调用LLM生成复盘报告
    final_prompt = summary_prompt.format(**summary_context)
    # 调用大模型生成追问问题
    llm = llms.get_llm_for_request(state.get("api_config"), channel="smart")
    feedback_report = await llm.ainvoke(final_prompt)
    logger.info(f"复盘报告生成完成 | 会话ID: {state['session_id']}")
    # 4. 更新状态
    updated_state = {
        "interview_status": "completed",
        "interview_result": feedback_report.content
    }
    return updated_state

async def start_route(state: InterviewState):
    """
        根据是否有用户回答的问题路由到初始化节点或者面试执行节点
    """
    current_answer = state.get("current_answer", None)
    if current_answer:
        return "execute_interview"
    else:
        return "init_interview"

async def execute_route(state: InterviewState):
    """
        判断直接结束等待用户回答，或者用户已回答走下一轮
    """
    need_user_answer = state.get("need_user_answer", False)
    current_answer = state.get("current_answer", False)
    if not need_user_answer and current_answer:
        return "react_decision"
    else:
        return END

async def react_route(state: InterviewState):
    """
        根据react_decision决定路由到哪里
    """
    decision = state.get("react_decision", "end")
    if decision in ("followup", "switch_topic", "switch_type"):
        return decision
    return "end"

async def build_interview_graph():
    """
    构建面试图谱
    """
    workflow = StateGraph(InterviewState)

    # 添加节点
    workflow.add_node("init_interview", init_interview)
    workflow.add_node("rag_retrieve", rag_retrieve)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("execute_interview", execute_interview)
    workflow.add_node("react_decision", react_decision)
    workflow.add_node("score_evaluation", score_evaluation)
    workflow.add_node("summary_feedback", summary_feedback)

    # 设置入口
    workflow.set_conditional_entry_point(
        start_route,
        {
            "init_interview": "init_interview",
            "execute_interview": "execute_interview"
        }
    )
    workflow.add_edge("init_interview", "rag_retrieve")
    workflow.add_edge("rag_retrieve", "generate_question")
    workflow.add_edge("generate_question", "execute_interview")
    workflow.add_edge("score_evaluation", "summary_feedback")
    workflow.add_edge("summary_feedback", END)

    workflow.add_conditional_edges(
        "execute_interview",
        execute_route,
        {
            END: END,
            "react_decision": "react_decision"
        }
    )

    workflow.add_conditional_edges(
        "react_decision",
        react_route,
        {
            "followup": "generate_question",
            "switch_topic": "rag_retrieve",
            "end": "score_evaluation",
            "switch_type": "rag_retrieve"
        }
    )
    # 注册图实例
    checkpointer = await get_async_sqlite_saver()
    graph = workflow.compile(checkpointer=checkpointer)
    register_graph_instance(graph)
    return graph

_graph_instances = []

def register_graph_instance(graph):
    """注册图实例以便后续清理"""
    _graph_instances.append(graph)
    return graph

def get_graph_instances():
    """获取所有图实例"""
    return _graph_instances

def clear_graph_instances():
    """清空图实例列表"""
    _graph_instances.clear()
