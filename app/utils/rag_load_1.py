import os
import pickle

import chromadb
import jieba
import pdfplumber
import re
from typing import List, Dict
import json

from chromadb import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


# 初始化嵌入模型（用于生成向量）
def get_chroma_collect():
    VECTOR_DB_DIR = "../resources/chroma_native_db"
    collection_name = "rag_knowledge_base"
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    return chroma_client.get_or_create_collection(
        name=collection_name
    )
# ===================== 依赖注入 =====================
def get_chroma_embedding():
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # 完整模型名称（可简写为 all-MiniLM-L6-v2）
        model_kwargs={"device": "cpu"},  # 无GPU用cpu
        encode_kwargs={"normalize_embeddings": True}  # 归一化向量（提升检索效果）
    )
    return embedder
embedder = get_chroma_embedding()
# 1. 识别文件格式并读取PDF
def read_pdf_with_pdfplumber(file_path: str) -> str:
    """
    使用pdfplumber读取PDF文件，提取所有文本
    """
    all_text = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 提取页面文本
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n\n".join(all_text)


# 2. 清理文本内容
def clean_text_content(text: str) -> str:
    """
    清理提取的文本内容
    """
    # 移除多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', text)

    # 标准化空格
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)

    # 移除PDF解析可能产生的特殊字符
    cleaned = re.sub(r'\x0c', '', cleaned)  # 换页符

    # 确保中文字符和标点的连续性
    cleaned = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', cleaned)

    return cleaned.strip()


# 3. 按照问题进行分块
def chunk_by_questions(text: str) -> List[Dict]:
    """
    按照问题进行分块，每个问题分一块
    """
    chunks = []
    current_chunk = []
    current_title = ""

    # 定义问题标题模式
    # 匹配中文数字编号：一、二、...、七、
    cn_num_pattern = r'^[一二三四五六七八九十]、'
    # 匹配数字编号：2.1, 5.3, 6.1等
    num_pattern = r'^\d+\.\d+'
    # 匹配带#的标题
    hash_pattern = r'^#+\s*[一二三四五六七八九十]、'

    # 按行分割文本
    lines = text.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 检查是否为问题标题
        is_title = False

        # 检查各种标题模式
        if (re.match(cn_num_pattern, line) or
                re.match(num_pattern, line) or
                re.match(hash_pattern, line) or
                re.match(r'^##\s+\d+\.', line) or  # 匹配 ## 2.1 等
                re.match(r'^##\s+[一二三四五六七八九十]、', line) or  # 匹配 ## 一、 等
                re.match(r'^\d+\.\d+\.\d+', line)):  # 匹配 2.1.1 等

            is_title = True

            # 如果当前块有内容，保存前一个块
            if current_chunk and current_title:
                chunk_content = '\n'.join(current_chunk).strip()
                if chunk_content:  # 确保块内容不为空
                    chunks.append('title'+current_title+'\ncontent'+chunk_content)

            # 开始新块
            current_title = line
            current_chunk = []

        # 如果不是标题，添加到当前块
        if not is_title and current_title:
            current_chunk.append(line)

    # 添加最后一个块
    if current_chunk and current_title:
        chunk_content = '\n'.join(current_chunk).strip()
        if chunk_content:
            chunks.append(
                'title'+current_title+'\ncontent'+chunk_content
            )

    return chunks


# 4. 保存到Chroma数据库
def save_to_chroma(chunks: List[str]):
    """
    将分块保存到Chroma向量数据库
    """
    collection = get_chroma_collect()
    # 生成嵌入向量
    ids = []
    for i, chunk in enumerate(chunks):
        ids.append(f"rag_1_{i}")
    embeddings = embedder.embed_documents(chunks)

    # 写入数据库
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    print(f"成功写入 {len(chunks)} 条文档到知识库")

    return collection


# 5. 完整的处理流程
def process_pdf_and_save_to_chroma(pdf_path: str):
    """
    完整的PDF处理流程：读取、清理、分块、保存到Chroma
    """
    print("开始处理PDF文件...")

    # 1. 读取PDF
    print("正在读取PDF文件...")
    raw_text = read_pdf_with_pdfplumber(pdf_path)
    print(f"原始文本长度: {len(raw_text)} 字符")

    # 2. 清理文本
    print("正在清理文本...")
    cleaned_text = clean_text_content(raw_text)
    print(f"清理后文本长度: {len(cleaned_text)} 字符")

    # 3. 按问题分块
    print("正在按问题分块...")
    chunks = chunk_by_questions(cleaned_text)
    print(f"共生成 {len(chunks)} 个问题块")
    return chunks

# 7. 保存分块到文本文件（用于调试）
def save_chunks_to_file(chunks: List[Dict], output_file: str):
    """
    保存分块到文本文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== 块 {i + 1} ===\n")
            f.write(f"标题: {chunk['title']}\n")
            f.write(f"内容:\n{chunk['content']}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"分块已保存到: {output_file}")


def query_db(query_text, n_results=3):
    """
    查询数据库
    :param query_text: 查询文本
    :param n_results: 返回结果数量
    :return: 查询结果列表
    """
    # 生成查询向量
    query_embedding = embedder.embed_documents([query_text])
    collection = get_chroma_collect()
    # 执行查询
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    # 格式化结果
    formatted_results = []
    for i in range(len(results["ids"][0])):
        formatted_results.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })

    return formatted_results


BM25_INDEX_PATH = "../resources/bm25_index.pkl"
CORPUS_PATH = "../resources/corpus.pkl"


def chinese_word_segment(text: str) -> List[str]:
    """
    去停用词 + 过滤无意义词 + 去重 + 长度过滤
    :param text: 待分词文本（原始考点/假设文档/任意检索文本）
    :return: 去重后的核心技术关键词列表
    """
    # ========== 1. 文本预处理：保留中文、英文、数字，过滤所有标点/特殊符号，通用预处理规则 ==========
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", str(text)).strip()
    if not text:
        return []

    # ========== 2. jieba 核心分词：精准模式（技术文本首选，无冗余分词，速度快） ==========
    # cut_all=False → 精准模式，适合技术文档/面试考点，不会把 "向量检索" 拆成 "向量" "检索"
    seg_list = jieba.cut(text, cut_all=False)

    # ========== 3. 超全停用词表：技术面试场景专属，过滤所有无意义词汇，只留核心技术词 ==========
    stop_words = {
        # 通用中文停用词
        "的", "了", "是", "在", "和", "与", "及", "等", "有", "就", "都", "也", "还", "为", "对", "此", "该", "即", "则", "若", "如",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "个", "本", "这", "那", "哪", "每", "各", "又", "而", "于", "因",
        # 技术文本无意义词
        "进行", "实现", "使用", "采用", "通过", "结合", "基于", "包含", "涉及", "需要", "可以", "能够", "应该", "必须", "建议",
        "问题", "答案", "解析", "考点", "面试", "岗位", "项目", "开发", "工程", "应用", "系统", "框架", "模块", "功能"
    }

    # ========== 4. 过滤逻辑：去停用词 + 过滤超短词 + 过滤纯数字 ==========
    core_words = []
    for word in seg_list:
        # 过滤规则1：不在停用词表内
        # 过滤规则2：词长度 ≥2 （单字无意义，技术词都是≥2）
        # 过滤规则3：不是纯数字（如 2024/100 这类无意义）
        if word not in stop_words and len(word) >= 2 and not word.isdigit():
            core_words.append(word)

    core_words = list(set(core_words))

    return core_words

def saveBm25(chunks: List[Dict]):
    docs = []
    for chunk in chunks:
        docs.append(chunk['title'] + "\n\n" + chunk['content'])

    # 1. 加载旧语料和旧BM25索引（若无则初始化空）
    old_corpus = []
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "rb") as f:
            old_corpus = pickle.load(f)

    # 2. 合并新旧语料（可选去重，避免重复文档）
    # 去重：利用集合去重（若文档是字符串），保持顺序
    merged_corpus = old_corpus + [doc for doc in docs if doc not in old_corpus]
    tokenized_corpus = [chinese_word_segment(doc) for doc in merged_corpus]
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized_corpus)

    # 3. 持久化BM25和语料
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(docs, f)

def query(query: str):

    bm25 = None
    corpus = []
    if os.path.exists(BM25_INDEX_PATH) and os.path.exists(CORPUS_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
    tokenized_query = chinese_word_segment(query)
    bm25_scores = bm25.get_scores(tokenized_query) if bm25 else []
    # 取BM25 Top-K结果
    bm25_results = []
    if len(bm25_scores) > 0:
        bm25_top_idx = bm25_scores.argsort()[-5:][::-1]
        bm25_results = [
            (corpus[idx], bm25_scores[idx])
            for idx in bm25_top_idx if bm25_scores[idx] > 0
        ]
    return bm25_results
# 主函数
def main():
    # PDF文件路径（请替换为实际路径）
    pdf_path = "../resources/大模型 RAG 经验面.pdf"  # 如果文件在当前目录

    # 处理PDF并保存到Chroma
    result = process_pdf_and_save_to_chroma(pdf_path)
    saveBm25(result)
    print(query("如何提升检索效果"))
    # 保存分块到文本文件
    save_chunks_to_file(result, "分块结果.txt")
    # 保存到向量数据库
    save_to_chroma(result)

    print(query_db("如何构建rag索引？", n_results=3))

    # 示例查询
    print("\n" + "=" * 60)


# 直接运行此脚本
if __name__ == "__main__":
    main()
    # embedder = get_chroma_embedding()
