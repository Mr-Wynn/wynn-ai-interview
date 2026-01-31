#!/usr/bin/env python3
from app.database.config import DatabaseConnector

# 建表SQL（字段类型严格匹配，避免类型错误）
CREATE_TABLE_SQLS = [
    # 1. 简历表（独立存储）
    """
    CREATE TABLE IF NOT EXISTS resume (
        id INT AUTO_INCREMENT PRIMARY KEY COMMENT '简历ID',
        resume_path VARCHAR(512) NOT NULL UNIQUE COMMENT '简历本地路径',
        file_name VARCHAR(256) NOT NULL COMMENT '简历原文件名',
        upload_time BIGINT NOT NULL COMMENT '上传时间戳（long）',
        user_id VARCHAR(64) DEFAULT 'default_user' COMMENT '用户ID'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='简历信息表';
    """,
    # 2. 面试会话表（与状态解耦，无外键）
    """
    CREATE TABLE IF NOT EXISTS interview_session (
        session_id VARCHAR(64) PRIMARY KEY COMMENT '会话ID',
        resume_id INT NOT NULL COMMENT '关联简历ID（无外键）',
        status VARCHAR(32) NOT NULL DEFAULT 'in_progress' COMMENT '会话状态',
        create_time BIGINT NOT NULL COMMENT '创建时间戳',
        update_time BIGINT NOT NULL COMMENT '更新时间戳',
        job_description TEXT NOT NULL COMMENT '岗位描述',
        company_info VARCHAR(1024) DEFAULT '' COMMENT '公司信息',
        interview_focus VARCHAR(32) NOT NULL DEFAULT 'balanced' COMMENT '面试侧重'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='面试会话表';
    """,
    # 3. 面试状态表（完全拆分InterviewState字段，无大JSON）
    """
    CREATE TABLE IF NOT EXISTS interview_state (
        id INT AUTO_INCREMENT PRIMARY KEY COMMENT '状态记录ID',
        session_id VARCHAR(64) NOT NULL COMMENT '关联会话ID',
        resume_path VARCHAR(512) NOT NULL COMMENT '简历路径',
        job_description VARCHAR(1024) NOT NULL COMMENT '岗位描述',
        company_info VARCHAR(1024) DEFAULT '' COMMENT '公司信息',
        interview_focus VARCHAR(32) NOT NULL DEFAULT 'balanced' COMMENT '面试侧重',
        api_config JSON COMMENT '大模型配置',
        resume_projects JSON COMMENT '解析的项目列表',
        project_tech_points JSON COMMENT '项目技术点',
        general_tech_points JSON COMMENT '通用考点',
        current_exam_point_type VARCHAR(32) COMMENT '当前题型类型',
        resume_text TEXT COMMENT '解析的简历文本',
        interview_status VARCHAR(32) DEFAULT 'in_progress' COMMENT '面试状态',
        rag_results JSON COMMENT 'RAG检索结果',
        current_exam_point VARCHAR(128) COMMENT '当前考点',
        current_question TEXT COMMENT '当前问题',
        need_user_answer BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否需要用户回答',
        current_answer TEXT COMMENT '用户当前回答',
        answer_quality TEXT COMMENT '回答质量',
        current_point_history JSON COMMENT '当前考点历史',
        history JSON COMMENT '总历史对话',
        completed_points JSON COMMENT '已完成考点',
        react_decision VARCHAR(32) COMMENT 'ReAct决策',
        question_count INT NOT NULL DEFAULT 0 COMMENT '已生成问题数',
        follow_up_reason VARCHAR(1024) COMMENT '追问原因',
        weak_points VARCHAR(1024) COMMENT '薄弱点',
        interview_result VARCHAR(1024) COMMENT '面试结果',
        score_details VARCHAR(1024) COMMENT '评分详情',
        max_questions INT NOT NULL DEFAULT 10 COMMENT '最大问题数',
        follow_up_count INT NOT NULL DEFAULT 0 COMMENT '当前追问次数',
        max_follow_ups INT NOT NULL DEFAULT 3 COMMENT '最大追问次数',
        create_time BIGINT NOT NULL COMMENT '创建时间戳',
        update_time BIGINT NOT NULL COMMENT '更新时间戳',
        INDEX idx_session_id (session_id) COMMENT '会话ID索引'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='面试状态表';
    """
]

def init_tables():
    """初始化所有表（纯原生SQL，无ORM，规避InstrumentedAttribute）"""
    db = DatabaseConnector()

    print("开始初始化数据库表...")
    for idx, sql in enumerate(CREATE_TABLE_SQLS, 1):
        db.execute_sql(sql)
        print(f"第{idx}张表创建/检查完成")
    print("🎉 所有表初始化成功！")


if __name__ == "__main__":
    # 手动执行：python -m app.database.init_tables
    init_tables()