import pymysql
import json
from typing import Optional, List, Dict, Any

# 数据库配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "xxxx",
    "database": "ai_interview",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor  # 返回字典格式结果
}

class DatabaseConnector:
    """纯原生数据库连接器，避免ORM相关问题"""
    def __init__(self):
        self.conn: Optional[pymysql.Connection] = None
        self.cursor: Optional[pymysql.cursors.DictCursor] = None

    def connect(self):
        """建立数据库连接"""
        if not self.conn or not self.conn.open:
            self.conn = pymysql.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
        return self

    def close(self):
        """关闭连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn and self.conn.open:
            self.conn.close()

    def execute_sql(self, sql: str, params: Optional[List[Any]] = None) -> Optional[List[Dict]]:
        """
        执行SQL语句
        :param sql: 原生SQL语句（使用%s作为占位符）
        :param params: SQL参数列表（防注入，避免字符串拼接）
        :return: 查询结果（SELECT）/None（INSERT/UPDATE/DELETE）
        """
        try:
            self.connect()
            # 执行SQL，参数必须是列表/元组，避免InstrumentedAttribute
            self.cursor.execute(sql, params or ())
            self.conn.commit()
            # 获取查询结果（SELECT）
            if sql.strip().upper().startswith("SELECT"):
                return self.cursor.fetchall()
            return None
        except Exception as e:
            self.conn.rollback()
            raise Exception(f"SQL执行失败: {str(e)} | SQL: {sql} | Params: {params}")

    def get_one(self, sql: str, params: Optional[List[Any]] = None) -> Optional[Dict]:
        """执行查询并返回单条结果"""
        results = self.execute_sql(sql, params)
        return results[0] if results else None

    # JSON字段序列化/反序列化工具（解决列表/字典存储）
    @staticmethod
    def serialize_json(obj: Any) -> str:
        """序列化Python对象为JSON字符串"""
        return json.dumps(obj, ensure_ascii=False) if obj else "[]" if isinstance(obj, list) else "{}"

    @staticmethod
    def deserialize_json(s: str) -> Any:
        """反序列化JSON字符串为Python对象"""
        if not s or s in ("[]", "{}"):
            return [] if "[" in s else {}
        return json.loads(s)