import logging
import os
import pickle
import re

import chromadb
import jieba
import numpy as np
from typing import List, Dict
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from app.core import llms
from app.utils.response_util import parse_response_list

# 日志配置
logger = logging.getLogger(__name__)

# ===================== 全局配置 =====================
def get_chroma_collect():
    VECTOR_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/chroma_native_db"))
    logger.info(f"VECTOR_DB_DIR is {VECTOR_DB_DIR}")
    collection_name = "rag_knowledge_base"
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "RAG面试经验文档"}
    )

def getBM25():
    BM25_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/bm25_index.pkl"))
    CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/corpus.pkl"))
    bm25 = None
    corpus = []
    if os.path.exists(BM25_INDEX_PATH) and os.path.exists(CORPUS_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25 = pickle.load(f)
        with open(CORPUS_PATH, "rb") as f:
            corpus = pickle.load(f)
    return bm25, corpus

KNOWLEDGE_COLLECTION = get_chroma_collect()
BM25, CORPUS = getBM25()

# 核心配置
SINGLE_RETRIEVE_TOP_K = 10    # 单个Query检索召回数
FINAL_RETRIEVE_TOP_K = 12     # 合并后最终保留数
DUPLICATE_SIM_THRESHOLD = 0.8 # 语义去重阈值
# 重排权重（仅检索相关性维度）
RERANK_WEIGHTS = {
    "vector_similarity": 0.55,
    "keyword_match_score": 0.25,
    "retrieve_source": 0.2
}

# ===================== 依赖注入 =====================
def get_chroma_embedding() -> Embeddings:
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # 完整模型名称（可简写为 all-MiniLM-L6-v2）
        model_kwargs={"device": "cpu"},  # 无GPU用cpu
        encode_kwargs={"normalize_embeddings": True}  # 归一化向量（提升检索效果）
    )
    return embedder

EMBEDDING_MODEL = get_chroma_embedding()

# ===================== 核心工具函数 =====================
def get_text_embedding(text: str) -> np.ndarray:
    # all-MiniLM-L6-v2 原生支持直接生成向量，无需reshape，兼容原返回格式
    return np.array(EMBEDDING_MODEL.embed_documents([text])).reshape(1, -1)

def calculate_cosine_sim(text1: str, text2: str) -> float:
    return cosine_similarity(get_text_embedding(text1), get_text_embedding(text2))[0][0]

def load_prompt_template(single_topic, generate_num, job_description, current_exam_point_type):
    hyde_param = {"single_topic":single_topic, "generate_num":generate_num, "job_description":job_description, "current_exam_point_type":current_exam_point_type}
    prompt = PromptTemplate.from_file("resources/hyde_query_prompt.txt", "utf-8")
    final_prompt = prompt.format(**hyde_param)
    return final_prompt

def min_max_normalize(data: np.ndarray):
    return (data - data.min()) / (data.max() - data.min())

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

async def hyde_generate_hypo_docs(single_topic: str, state) -> list:
    """
    生成3个不同维度的假设文档，覆盖考点的全场景语义
    """
    llm = llms.get_llm_for_request(state.get("api_config"), channel="fast")
    job_desc = state.get("job_description", "LLM应用开发工程师")
    topic_type = state.get("current_exam_point_type", "general")
    hyde_prompt = load_prompt_template(single_topic, 3, job_desc, topic_type)
    hyde_response = await llm.ainvoke(hyde_prompt)
    hydes = parse_response_list(hyde_response.content.strip())
    hydes.append(single_topic)
    return hydes

def single_query_retrieve(query_type: str, query_content: str) -> List[Dict]:
    """
    【全新重构，核心优化】单Query的混合检索逻辑 - 彻底解决你的2个核心问题
    规则：对当前Query，执行【专属向量检索】+【专属关键词检索】，两者数据源完全匹配当前Query
    1. 向量检索：使用 当前Query的完整文本 → 做语义召回
    2. 关键词检索：使用 当前Query分词后的核心词集 → 做精准召回
    """
    vector_results = KNOWLEDGE_COLLECTION.query(
        query_embeddings=get_text_embedding(query_content).tolist(),
        n_results=SINGLE_RETRIEVE_TOP_K,
        include=["documents", "distances"]
    )
    vector_docs = [
        {
            "content": doc,
            "vector_score": 1 - dist,
            "source": "vector",
            "query_type": query_type,
            "query_keywords": chinese_word_segment(query_content) # 新增：当前Query的分词关键词
        }
        for doc, dist in zip(vector_results["documents"][0], vector_results["distances"][0])
    ]

    # 对当前Query做分词
    current_query_keywords = chinese_word_segment(query_content)
    bm25_scores = BM25.get_scores(current_query_keywords) if BM25 else []
    # 取BM25 Top-K结果
    keyword_docs = []
    if len(bm25_scores) > 0:
        bm25_top_idx = bm25_scores.argsort()[-SINGLE_RETRIEVE_TOP_K:][::-1]
        keyword_docs = [
            {
                "content": CORPUS[idx],
                "vector_score": bm25_scores[idx],
                "source": "keyword",
                "query_type": query_type,
                "query_keywords": current_query_keywords
            }
            for idx in bm25_top_idx if bm25_scores[idx] > 0
        ]

    # ========== 3. 单Query结果去重（避免同Query内重复） ==========
    combined_docs = vector_docs + keyword_docs
    seen_content = set()
    unique_docs = []
    for doc in combined_docs:
        clean_content = doc["content"].strip()
        if clean_content not in seen_content and len(clean_content) > 20:
            seen_content.add(clean_content)
            unique_docs.append(doc)
    logger.info(f"单Query检索完成 [{query_type}]，分词关键词：{current_query_keywords}，召回{len(unique_docs)}条素材")
    return unique_docs

def merge_and_rerank_all_docs(all_docs: List[Dict]) -> List[Dict]:
    """
    全局结果合并+相关性重排
    """
    vector_scores=[]
    keyword_scores=[]
    source_weights=[]
    for doc in all_docs:
        # 1. 向量相似度得分
        vector_score = doc["vector_score"]
        vector_scores.append(vector_score)
        # 2. 基于当前Query的分词关键词集计算命中数
        content_lower = doc["content"].lower()
        query_keywords = doc["query_keywords"]
        keyword_hit_count = sum([1 for kw in query_keywords if kw.lower() in content_lower])
        keyword_scores.append(keyword_hit_count)
        # 3. 检索来源权重
        source_weight = 0.5 if doc["source"] == "vector" else 0.3
        source_weights.append(source_weight)
    vector_scores = min_max_normalize(np.array(vector_scores))
    keyword_scores = min_max_normalize(np.array(keyword_scores))
    source_weights = min_max_normalize(np.array(source_weights))
    for i, doc in enumerate(all_docs):
        doc["global_rank_score"] = (
                vector_scores[i] * RERANK_WEIGHTS["vector_similarity"]
                + keyword_scores[i] * RERANK_WEIGHTS["keyword_match_score"]
                + source_weights[i] * RERANK_WEIGHTS["retrieve_source"]
        )

    # 全局排序 + 截断
    sorted_docs = sorted(all_docs, key=lambda x: x["global_rank_score"], reverse=True)
    return sorted_docs[:FINAL_RETRIEVE_TOP_K]

def global_semantic_deduplicate(rerank_docs: List[Dict]) -> List[str]:
    """
    规则：重排后文档已按相关性排序，先保留的文档相关性更高，相似文档直接跳过
    """
    final_unique_contents = []
    for doc in rerank_docs:
        current_content = doc["content"].strip()
        if not current_content:
            continue
        # 判断是否与已保留内容相似
        is_duplicate = False
        for exist_content in final_unique_contents:
            sim_score = calculate_cosine_sim(current_content, exist_content)
            if sim_score >= DUPLICATE_SIM_THRESHOLD:
                is_duplicate = True
                logger.debug(f"全局去重：跳过相似内容 [{doc['query_type']}] {current_content[:30]}...")
                break
        if not is_duplicate:
            final_unique_contents.append(current_content)
    return final_unique_contents

# ===================== 对外核心入口 =====================
async def rag_retrieve_core(single_topic: str, state) -> List[str]:
    """
    最终完整流程：
    1. 生成3个假设文档 → 构建4个独立Query
    2. 4个Query分别执行混合检索
    3. 合并所有结果 → 全局相关性重排
    4. 全局纯语义去重 → 返回最终结果
    """
    logger.info(f"开始执行Query HYDE混合检索，目标考点: [{single_topic}]")
    # 步骤1：构建4个独立检索Query
    hyde_queries = await hyde_generate_hypo_docs(single_topic, state)

    # 步骤2：4个Query分别检索，收集所有结果
    all_retrieve_docs = []
    for query_content in hyde_queries:
        single_docs = single_query_retrieve(query_content, single_topic)
        all_retrieve_docs.extend(single_docs)
    if not all_retrieve_docs:
        logger.warning(f"考点[{single_topic}] 4Query检索无结果")
        return []
    # 步骤3：全局结果合并+相关性重排
    global_rerank_docs = merge_and_rerank_all_docs(all_retrieve_docs)
    # 步骤4：全局纯语义去重
    final_contents = global_semantic_deduplicate(global_rerank_docs)
    # 日志输出
    logger.info(f"考点[{single_topic}] 4Query检索完成，最终返回 {len(final_contents)} 条高相关无重复素材")
    return final_contents


