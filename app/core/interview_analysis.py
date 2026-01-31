
import logging
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

from app.core import llms
from app.utils.response_util import parse_response_dict

logger = logging.getLogger(__name__)



async def extract_resume_text(resume_path: str):
    """提取简历文本，支持PDF和纯文本"""
    if not resume_path:
        raise Exception(f"PDF解析失败, resume_path为空: {resume_path}")

    # 检查是否是PDF文件路径
    if os.path.exists(resume_path) and resume_path.endswith('.pdf'):
        try:
            loader = PyPDFLoader(resume_path)
            documents = loader.load()
            text = "\n\n".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            logger.error(f"PDF解析失败，文件路径不正确: {resume_path}", e)
            raise e

    # 检查是否是Base64编码的PDF
    if len(resume_path) > 100 and "JVBER" in resume_path[:100]:
        try:
            import base64
            pdf_data = base64.b64decode(resume_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_data)
                pdf_path = tmp_file.name

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text = "\n\n".join([doc.page_content for doc in documents])

            os.unlink(pdf_path)
            return text
        except Exception as e:
            logger.error(f"Base64 PDF解析失败: {resume_path}", e)
            raise e
    raise Exception(f"PDF解析失败: {resume_path}")

def load_prompt_template(analysis_context: dict):
    # ["resume_context", "job_description", "company_info", "interview_focus", "general_topics_str"]
    prompt = PromptTemplate.from_file("resources/resume_analysis_prompt.txt", "utf-8")
    final_prompt = prompt.format(**analysis_context)
    return final_prompt

async def analyze_resume_with_llm(
        analysis_context: dict,
        api_config: dict
) -> dict:
    """
    使用大模型分析简历，生成考点计划
    """
    # 获取大模型配置
    llm = llms.get_llm_for_request(api_config, channel="fast")
    # 加载提示词模板
    prompt_template_str = load_prompt_template(analysis_context)
    try:
        # 调用大模型
        response = await llm.ainvoke(prompt_template_str)
        response_text = response.content

        # 解析JSON
        result = parse_response_dict(response_text)

        # 验证必要字段
        required_fields = ["resume_projects", "project_tech_points", "general_tech_points", "first_topic_type"]
        for field in required_fields:
            if field not in result:
                raise Exception(f"解析简历失败{response_text}")

        # 确保first_topic_type是有效的
        if result["first_topic_type"] not in ["general", "project"]:
            if analysis_context.get("interview_focus") == "project" and result["resume_projects"]:
                result["first_topic_type"] = "project"
            else:
                result["first_topic_type"] = "general"

        return result

    except Exception as e:
        logger.error(f"大模型分析失败:", e)
        raise Exception("解析简历失败", e)
