import json


def parse_response_dict(response_text: str) -> dict:
    # 尝试清理可能的 markdown 格式
    cleaned_text = parse_response_str(response_text)
    result = json.loads(cleaned_text)
    return result

def parse_response_list(response_text: str) -> list:
    # 尝试清理可能的 markdown 格式
    cleaned_text = parse_response_str(response_text)
    result = json.loads(cleaned_text)
    return result

def parse_response_str(response_text: str) -> str:
    """
    解析 LLM 返回的面试计划 JSON

    Args:
        response_text: LLM 返回的原始文本
    Returns:
        面试问题列表
    """
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    return cleaned_text