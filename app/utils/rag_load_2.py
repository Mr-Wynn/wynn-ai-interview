import re
from typing import List

import chromadb

from app.utils.rag_load_1 import get_chroma_collect, save_to_chroma, query_db


def load_markdown_file(file_path):
    """加载Markdown文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def extract_questions_and_subquestions(content) -> List[str]:
    """从Markdown内容中提取问题和子问题"""
    chunks = []

    # 移除目录部分（直到第一个<h2>标签之前的所有内容）
    content = re.sub(r'^.*?(?=<h2)', '', content, flags=re.DOTALL)

    # 先找到所有的h2标题块
    h2_pattern = r'<h2 id="[^"]*">([^<]+)</h2>(.*?)(?=<h2 id="|$)'
    h2_matches = re.findall(h2_pattern, content, re.DOTALL)

    for main_title, main_content in h2_matches:
        # 清理主标题
        clean_main_title = main_title.strip()

        # 提取h3级别的子问题
        # 匹配 ### 标题或加粗的标题（如 **1. **角色定义与人格建模**）
        sub_pattern = r'(?:###\s+([^\n]+)|####\s+\d+\.\s+\*\*([^\*]+)\*\*)\n(.*?)(?=(?:###|####|\Z))'
        sub_matches = re.findall(sub_pattern, main_content, re.DOTALL)

        if sub_matches:
            # 有子问题的情况
            for sub_match in sub_matches:
                # 子标题可能来自###格式或####加粗格式
                sub_title = sub_match[0] if sub_match[0] else sub_match[1]
                sub_content = sub_match[2]

                # 清理内容
                clean_sub_title = sub_title.strip()
                clean_sub_content = sub_content.strip()

                # 合并主标题和子标题作为文档标题，然后加上内容
                full_content = f"{clean_main_title}\n\n{clean_sub_title}\n\n{clean_sub_content}"
                chunks.append(full_content)
        else:
            # 没有子问题，整个h2内容作为一块
            clean_main_content = main_content.strip()
            full_content = f"{clean_main_title}\n\n{clean_main_content}"
            chunks.append(full_content)

    return chunks

def main():
    """主函数：加载、清理、分块、保存"""

    # 1. 加载文件
    file_path = "../resources/Agent基础知识.md"
    content = load_markdown_file(file_path)
    print(f"已加载文件: {file_path}")
    print(f"文件大小: {len(content)} 字符")

    # 2. 提取问题和子问题（分块）
    chunks = extract_questions_and_subquestions(content)
    print(f"提取到 {len(chunks)} 个分块")

    # 显示前几个分块的预览
    print("\n分块预览:")
    for i1, chunk in enumerate(chunks[:3]):
        print(f"\n--- 分块 {i1 + 1} (前200字符) ---")
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(preview)

    # 3. 保存到Chroma
    collection = save_to_chroma(chunks)

    # 4. 验证保存结果
    count = collection.count()
    print(f"验证：数据库中现有 {count} 个文档")

    return chunks


def show_all_chunks_in_chroma(collection_name="ai_agent_qa"):
    """显示Chroma中所有的分块"""

    try:
        # 初始化客户端
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # 获取集合
        collection = chroma_client.get_collection(name=collection_name)

        # 获取所有文档
        all_docs = collection.get()

        print(f"\nChroma中所有文档:")
        print(f"总数: {len(all_docs['ids'])}")

        for i, (doc_id, doc_content) in enumerate(zip(all_docs['ids'], all_docs['documents'])):
            print(f"\n--- 文档 {i + 1}: {doc_id} ---")
            # 只显示前100个字符作为预览
            preview = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
            print(preview)

        return all_docs
    except Exception as e:
        print(f"获取文档失败: {e}")
        return None


# 如果需要直接运行，取消下面的注释
if __name__ == "__main__":
    # 执行主流程
    chunks_res = main()

    # 显示Chroma中所有分块
    all_docs = show_all_chunks_in_chroma()

    # 示例查询
    print("\n" + "=" * 50)
    print("测试查询功能:")

    # 查询示例
    test_queries = ["function call", "上下文工程", "记忆机制"]

    for query in test_queries:
        print(f"\n查询 '{query}' 的结果:")
        results = query_db(query, n_results=2)

        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                print(f"\n--- 结果 {i + 1} ---")
                # 只显示前150个字符作为预览
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                print(preview)
        else:
            print("  无结果")