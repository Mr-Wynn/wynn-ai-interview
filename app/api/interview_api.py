import json
import logging
import os
import uuid
import time
from fastapi import UploadFile, File, HTTPException, APIRouter

from app.core.config import api_config
from app.database.db_service import db_service
from app.models.schemas import (
    ResumeUploadResponse, StartInterviewRequest, StartInterviewResponse,
    AnswerQuestionRequest, AnswerQuestionResponse, InterviewResultResponse
)
from app.core.react_graph import build_interview_graph, InterviewState
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/interview", tags=["聊天"])
# ===================== 核心接口 =====================
@router.post("/resume/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...), user_id: str = "default_user"):
    """1. 上传简历（保存文件+记录路径到数据库）"""
    # 生成唯一文件名，避免重复
    from app.main import RESUME_DIR
    file_suffix = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_suffix}"
    resume_path = os.path.join(RESUME_DIR, unique_filename)

    # 保存文件到本地
    with open(resume_path, "wb") as f:
        f.write(await file.read())

    # 调用数据库服务插入简历（无SQL）
    upload_ts = int(time.time())
    resume_id = db_service.insert_resume(resume_path, file.filename, upload_ts, user_id)
    if resume_id == -1:
        raise HTTPException(status_code=500, detail="简历记录插入失败")

    return ResumeUploadResponse(
        resume_id=resume_id,
        resume_path=resume_path
    )


@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(req: StartInterviewRequest):
    """2. 开始面试（创建session+初始化状态+生成第一个问题）"""
    # 1. 调用数据库服务校验简历存在
    resume_path = db_service.get_resume_path_by_id(req.resume_id)
    if not resume_path:
        raise HTTPException(status_code=404, detail="简历不存在")

    # 2. 调用数据库服务创建session（无SQL）
    session_id = str(uuid.uuid4())
    create_ts = int(time.time())
    db_service.insert_session(
        session_id=session_id,
        resume_id=req.resume_id,
        status="in_progress",
        create_time=create_ts,
        update_time=create_ts,
        job_description=req.job_description,
        company_info=req.company_info,
        interview_focus=req.interview_focus
    )

    # 3. 初始化InterviewState（匹配graph的状态结构）
    init_state: InterviewState = {
        "session_id": session_id,
        "resume_path": resume_path,
        "job_description": req.job_description,
        "company_info": req.company_info,
        "interview_focus": req.interview_focus,
        "api_config": api_config if not req.api_config else req.api_config,
        "resume_projects": [],
        "project_tech_points": [],
        "general_tech_points": [],
        "current_exam_point_type": None,
        "resume_text": None,
        "interview_status": "in_progress",
        "rag_results": [],
        "current_exam_point": None,
        "current_question": "",
        "need_user_answer": True,
        "current_answer": None,
        "answer_quality": None,
        "current_point_history": [],
        "history": [],
        "completed_points": [],
        "react_decision": None,
        "question_count": 0,
        "follow_up_reason": None,
        "weak_points": None,
        "interview_result": None,
        "score_details": None,
        "max_questions": 10,
        "follow_up_count": 0,
        "max_follow_ups": 3
    }

    # 4. 运行langgraph生成第一个问题
    graph = await build_interview_graph()
    result_state = await graph.ainvoke(init_state, config={"configurable": {"thread_id": session_id}})
    
    # 5. 调用数据库服务保存初始状态
    serialize = db_service.serialize_json
    state_params = [
        session_id,
        result_state["resume_path"],
        result_state["job_description"],
        result_state["company_info"],
        result_state["interview_focus"],
        serialize(result_state["api_config"]),
        serialize(result_state["resume_projects"]),
        serialize(result_state["project_tech_points"]),
        serialize(result_state["general_tech_points"]),
        result_state["current_exam_point_type"],
        result_state["resume_text"],
        result_state["interview_status"],
        serialize(result_state["rag_results"]),
        result_state["current_exam_point"],
        result_state["current_question"],
        result_state["need_user_answer"],
        result_state["current_answer"],
        result_state["answer_quality"],
        serialize(result_state["current_point_history"]),
        serialize(result_state["history"]),
        serialize(result_state["completed_points"]),
        result_state["react_decision"],
        result_state["question_count"],
        result_state["follow_up_reason"],
        result_state["weak_points"],
        result_state["interview_result"],
        result_state["score_details"],
        result_state["max_questions"],
        result_state["follow_up_count"],
        result_state["max_follow_ups"],
        int(time.time()),
        int(time.time())
    ]
    db_service.insert_interview_state(state_params)
    logger.info("start interview state: " + str(state_params))
    status = "wait_answer" if result_state["interview_status"] != "completed" else "completed"
    db_service.update_session_status(session_id, int(time.time()), status)
    return StartInterviewResponse(
        session_id=session_id,
        current_question=result_state["current_question"]
    )


@router.post("/answer", response_model=AnswerQuestionResponse)
async def answer_question(req: AnswerQuestionRequest):
    """3. 回答问题（更新状态+生成下一个问题）"""
    # 1. 调用数据库服务校验session状态（无SQL）
    session_status = db_service.get_session_status(req.session_id)
    if not session_status:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session_status == "completed":
        raise HTTPException(status_code=400, detail="面试已结束，无法回答")

    # 2. 调用数据库服务获取最新面试状态
    latest_state = db_service.get_latest_interview_state(req.session_id)
    if not latest_state:
        raise HTTPException(status_code=404, detail="面试状态不存在")
   
    db_service.update_session_status(req.session_id, int(time.time()), "in_progress")
    # 3. 转换数据库数据到InterviewState（反序列化JSON字段）
    deserialize = db_service.deserialize_json
    current_state: InterviewState = {
        "session_id": latest_state["session_id"],
        "resume_path": latest_state["resume_path"],
        "job_description": latest_state["job_description"],
        "company_info": latest_state["company_info"],
        "interview_focus": latest_state["interview_focus"],
        "api_config": deserialize(latest_state["api_config"]),
        "resume_projects": deserialize(latest_state["resume_projects"]),
        "project_tech_points": deserialize(latest_state["project_tech_points"]),
        "general_tech_points": deserialize(latest_state["general_tech_points"]),
        "current_exam_point_type": latest_state["current_exam_point_type"],
        "resume_text": latest_state["resume_text"],
        "interview_status": latest_state["interview_status"],
        "rag_results": deserialize(latest_state["rag_results"]),
        "current_exam_point": latest_state["current_exam_point"],
        "current_question": latest_state["current_question"],
        "need_user_answer": False,  # 用户已回答
        "current_answer": req.current_answer,
        "answer_quality": latest_state["answer_quality"],
        "current_point_history": deserialize(latest_state["current_point_history"]),
        "history": deserialize(latest_state["history"]),
        "completed_points": deserialize(latest_state["completed_points"]),
        "react_decision": latest_state["react_decision"],
        "question_count": latest_state["question_count"],
        "follow_up_reason": latest_state["follow_up_reason"],
        "weak_points": latest_state["weak_points"],
        "interview_result": latest_state["interview_result"],
        "score_details": latest_state["score_details"],
        "max_questions": latest_state["max_questions"],
        "follow_up_count": latest_state["follow_up_count"],
        "max_follow_ups": latest_state["max_follow_ups"]
    }

    # 4. 运行langgraph生成下一个问题
    graph = await build_interview_graph()
    result_state = await graph.ainvoke(current_state, config={"configurable": {"thread_id": req.session_id}})

    # 5. 检查面试是否结束，更新session状态
    status = "wait_answer" if result_state["interview_status"] != "completed" else "completed"
    db_service.update_session_status(req.session_id, int(time.time()), status)
    logger.info("update state is " + str(result_state))
    # 6. 调用数据库服务保存更新后的状态
    serialize = db_service.serialize_json
    state_params = [
        req.session_id,
        result_state["resume_path"],
        result_state["job_description"],
        result_state["company_info"],
        result_state["interview_focus"],
        serialize(result_state["api_config"]),
        serialize(result_state["resume_projects"]),
        serialize(result_state["project_tech_points"]),
        serialize(result_state["general_tech_points"]),
        result_state["current_exam_point_type"],
        result_state["resume_text"],
        result_state["interview_status"],
        serialize(result_state["rag_results"]),
        result_state["current_exam_point"],
        result_state["current_question"],
        result_state["need_user_answer"],
        result_state["current_answer"],
        result_state["answer_quality"],
        serialize(result_state["current_point_history"]),
        serialize(result_state["history"]),
        serialize(result_state["completed_points"]),
        result_state["react_decision"],
        result_state["question_count"],
        result_state["follow_up_reason"],
        result_state["weak_points"],
        result_state["interview_result"],
        result_state["score_details"],
        result_state["max_questions"],
        result_state["follow_up_count"],
        result_state["max_follow_ups"],
        int(time.time()),
        int(time.time())
    ]
    logger.info("update interview state: " + str(state_params))

    db_service.insert_interview_state(state_params)

    return AnswerQuestionResponse(
        session_id=req.session_id,
        current_question=result_state.get("current_question", ""),
        interview_completed=status == "completed",
        message="回答已提交" if not status == "completed" else "面试已结束"
    )


@router.get("/result/{session_id}", response_model=InterviewResultResponse)
async def get_interview_result(session_id: str):
    """4. 获取面试结果（仅面试结束后可查）"""
    # 1. 调用数据库服务校验session已结束（无SQL）
    session_status = db_service.get_session_status(session_id)
    if not session_status:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session_status != "completed":
        raise HTTPException(status_code=400, detail="面试未结束，无法获取结果")

    # 2. 调用数据库服务获取最新状态（无SQL）
    latest_state = db_service.get_latest_interview_state(session_id)
    if not latest_state or not latest_state["interview_result"]:
        raise HTTPException(status_code=404, detail="面试结果不存在")

    return InterviewResultResponse(
        session_id=session_id,
        interview_result=latest_state["interview_result"],
        weak_points=latest_state["weak_points"],
        score_details=latest_state["score_details"],
        history=db_service.deserialize_json(latest_state["history"])
    )
