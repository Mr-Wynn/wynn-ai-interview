from sqlalchemy import Column, String, BigInteger, JSON, Boolean, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
import time

Base = declarative_base()


# 简历表
class ResumeModel(Base):
    __tablename__ = "resume"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="简历ID")
    resume_path = Column(String(512), unique=True, nullable=False, comment="简历文件本地路径")
    file_name = Column(String(256), nullable=False, comment="简历原文件名")
    upload_time = Column(BigInteger, nullable=False, default=lambda: int(time.time()), comment="上传时间戳")
    user_id = Column(String(64), nullable=True, comment="用户ID（简化版）")


# 面试会话表
class InterviewSessionModel(Base):
    __tablename__ = "interview_session"

    session_id = Column(String(64), primary_key=True, comment="会话ID")
    resume_id = Column(Integer, nullable=False, comment="关联简历ID")
    status = Column(String(32), nullable=False, default="in_progress", comment="会话状态：in_progress/completed")
    create_time = Column(BigInteger, nullable=False, default=lambda: int(time.time()), comment="创建时间戳")
    update_time = Column(BigInteger, nullable=False, default=lambda: int(time.time()), comment="更新时间戳")
    job_description = Column(Text, nullable=False, comment="岗位描述")
    company_info = Column(Text, nullable=True, comment="公司信息")
    interview_focus = Column(String(32), nullable=False, default="balanced",
                             comment="面试侧重：project/general/balanced")


# 面试状态表
class InterviewStateModel(Base):
    __tablename__ = "interview_state"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="状态记录ID")
    session_id = Column(String(64), nullable=False, comment="关联会话ID")
    resume_path = Column(String(512), nullable=False, comment="简历路径")
    job_description = Column(Text, nullable=False, comment="岗位描述")
    company_info = Column(Text, nullable=True, comment="公司信息")
    interview_focus = Column(String(32), nullable=False, comment="面试侧重")
    api_config = Column(JSON, nullable=True, comment="大模型配置")
    resume_projects = Column(JSON, nullable=True, comment="解析的项目列表")
    project_tech_points = Column(JSON, nullable=True, comment="项目技术点")
    general_tech_points = Column(JSON, nullable=True, comment="通用考点")
    current_exam_point_type = Column(String(32), nullable=True, comment="当前题型类型")
    resume_text = Column(Text, nullable=True, comment="解析的简历文本")
    interview_status = Column(String(32), nullable=True, comment="面试状态：in_progress/completed")
    rag_results = Column(JSON, nullable=True, comment="RAG检索结果")
    current_exam_point = Column(String(128), nullable=True, comment="当前考点")
    current_question = Column(Text, nullable=True, comment="当前问题")
    need_user_answer = Column(Boolean, nullable=False, default=True, comment="是否需要用户回答")
    current_answer = Column(Text, nullable=True, comment="用户当前回答")
    answer_quality = Column(Text, nullable=True, comment="回答质量")
    current_point_history = Column(JSON, nullable=True, comment="当前考点历史对话")
    history = Column(JSON, nullable=True, comment="总历史对话")
    completed_points = Column(JSON, nullable=True, comment="已完成考点")
    react_decision = Column(String(32), nullable=True, comment="ReAct决策")
    question_count = Column(Integer, nullable=False, default=0, comment="已生成问题数")
    follow_up_reason = Column(Text, nullable=True, comment="追问原因")
    weak_points = Column(Text, nullable=True, comment="薄弱点")
    interview_result = Column(Text, nullable=True, comment="面试结果")
    score_details = Column(Text, nullable=True, comment="评分详情")
    max_questions = Column(Integer, nullable=False, default=10, comment="最大问题数")
    follow_up_count = Column(Integer, nullable=False, default=0, comment="当前追问次数")
    max_follow_ups = Column(Integer, nullable=False, default=3, comment="最大追问次数")
    create_time = Column(BigInteger, nullable=False, default=lambda: int(time.time()), comment="创建时间戳")
    update_time = Column(BigInteger, nullable=False, default=lambda: int(time.time()), comment="更新时间戳")