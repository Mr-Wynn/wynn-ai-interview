from typing import Optional, Dict, List, Any
from app.database.config import DatabaseConnector

"""所有SQL语句集中管理，API层不直接接触SQL"""

# ===================== 简历表相关SQL =====================
# 插入简历
RESUME_INSERT = """
    INSERT INTO resume (resume_path, file_name, upload_time, user_id)
    VALUES (%s, %s, %s, %s)
"""

# 根据路径查询简历
RESUME_SELECT_BY_PATH = "SELECT id FROM resume WHERE resume_path = %s LIMIT 1"

# 根据ID查询简历
RESUME_SELECT_BY_ID = "SELECT resume_path FROM resume WHERE id = %s LIMIT 1"

# ===================== 会话表相关SQL =====================
# 插入会话
SESSION_INSERT = """
    INSERT INTO interview_session 
    (session_id, resume_id, status, create_time, update_time, job_description, company_info, interview_focus)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

# 根据ID查询会话状态
SESSION_SELECT_STATUS = "SELECT status FROM interview_session WHERE session_id = %s LIMIT 1"

# 更新会话状态为已完成
SESSION_UPDATE_COMPLETED = """
    UPDATE interview_session 
    SET status = %s, update_time = %s 
    WHERE session_id = %s
"""

# ===================== 状态表相关SQL =====================
# 插入面试状态
STATE_INSERT = """
    INSERT INTO interview_state (
        session_id, resume_path, job_description, company_info, interview_focus,
        api_config, resume_projects, project_tech_points, general_tech_points,
        current_exam_point_type, resume_text, interview_status, rag_results,
        current_exam_point, current_question, need_user_answer, current_answer,
        answer_quality, current_point_history, history, completed_points,
        react_decision, question_count, follow_up_reason, weak_points,
        interview_result, score_details, max_questions, follow_up_count,
        max_follow_ups, create_time, update_time
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
"""

# 查询最新面试状态（按时间倒序）
STATE_SELECT_LATEST = """
    SELECT * FROM interview_state 
    WHERE session_id = %s 
    ORDER BY update_time DESC 
    LIMIT 1
"""

class InterviewDBService:
    """数据库操作封装服务，API层仅调用此类方法，不接触原始SQL"""
    def __init__(self):
        self.connector = DatabaseConnector()

    def close(self):
        self.connector.close()

    # ===================== 简历相关操作 =====================
    def insert_resume(self, resume_path: str, file_name: str, upload_time: int, user_id: str) -> int:
        """插入简历并返回简历ID"""
        self.connector.execute_sql(RESUME_INSERT, [resume_path, file_name, upload_time, user_id])
        resume_data = self.connector.get_one(RESUME_SELECT_BY_PATH, [resume_path])
        return resume_data["id"] if resume_data else -1

    def get_resume_path_by_id(self, resume_id: int) -> Optional[str]:
        """根据简历ID查询简历路径"""
        resume_data = self.connector.get_one(RESUME_SELECT_BY_ID, [resume_id])
        return resume_data["resume_path"] if resume_data else None

    # ===================== 会话相关操作 =====================
    def insert_session(
        self, session_id: str, resume_id: int, status: str, create_time: int,
        update_time: int, job_description: str, company_info: str, interview_focus: str
    ):
        """插入面试会话"""
        self.connector.execute_sql(SESSION_INSERT, [
            session_id, resume_id, status, create_time, update_time,
            job_description, company_info, interview_focus
        ])

    def get_session_status(self, session_id: str) -> Optional[str]:
        """查询会话状态"""
        session_data = self.connector.get_one(SESSION_SELECT_STATUS, [session_id])
        return session_data["status"] if session_data else None

    def update_session_status(self, session_id: str, update_time: int, status: str):
        """更新会话状态为已完成"""
        self.connector.execute_sql(SESSION_UPDATE_COMPLETED, [status, update_time, session_id])

    # ===================== 状态相关操作 =====================
    def insert_interview_state(self, state_params: List[Any]):
        """插入面试状态"""
        self.connector.execute_sql(STATE_INSERT, state_params)

    def get_latest_interview_state(self, session_id: str) -> Optional[Dict]:
        """获取最新的面试状态"""
        return self.connector.get_one(STATE_SELECT_LATEST, [session_id])

    # 工具方法：JSON序列化/反序列化
    @staticmethod
    def serialize_json(obj: Any) -> str:
        return DatabaseConnector.serialize_json(obj)

    @staticmethod
    def deserialize_json(s: str) -> Any:
        return DatabaseConnector.deserialize_json(s)

db_service = InterviewDBService()