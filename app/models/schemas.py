from pydantic import BaseModel, Field
from typing import Optional, List, Dict

# 1. 简历上传响应
class ResumeUploadResponse(BaseModel):
    resume_id: int
    resume_path: str
    message: str = "上传成功"

# 2. 开始面试请求
class StartInterviewRequest(BaseModel):
    resume_id: int
    job_description: str
    company_info: Optional[str] = ""
    interview_focus: str = Field(default="balanced", pattern="^(project|general|balanced)$")
    api_config: Optional[Dict] = Field(default=None)

# 3. 开始面试响应
class StartInterviewResponse(BaseModel):
    session_id: str
    current_question: str
    message: str = "面试已开始"

# 4. 回答问题请求
class AnswerQuestionRequest(BaseModel):
    session_id: str
    current_answer: str

# 5. 回答问题响应
class AnswerQuestionResponse(BaseModel):
    session_id: str
    current_question: Optional[str] = None
    interview_completed: bool = False
    message: str

# 6. 获取面试结果响应
class InterviewResultResponse(BaseModel):
    session_id: str
    interview_result: str
    weak_points: Optional[str]
    score_details: Optional[str]
    history: List[Dict]