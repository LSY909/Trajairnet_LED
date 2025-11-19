"""Time Series Retrieval Augmented Generation (RAG) Module.

This module provides functionality for storing and retrieving time series data using
vector similarity search. It implements a RAG system specifically designed for time
series data, allowing efficient storage and retrieval of similar patterns.

Example:
    # >>> from timeseries_rag.models import TimeSeriesEmbedder
    # >>> embedder = TimeSeriesEmbedder()
    # >>> rag = TimeSeriesRAG()
    # >>>
    # >>> # Add a document
    # >>> ts_data = np.sin(np.linspace(0, 10, 100))
    # >>> embedding = embedder.embed(ts_data)
    # >>> doc = TimeSeriesDocument(
    # ...     id="sin_wave_1",
    # ...     data=ts_data,
    # ...     metadata={"type": "sine", "frequency": 1.0},
    # ...     embedding=embedding
    # ... )
    # >>> rag.add_document(doc)
    # >>>
    # >>> # Search for similar patterns
    # >>> query = np.sin(np.linspace(0, 10, 100) + 0.1)
    # >>> query_embedding = embedder.embed(query)
    # >>> results = rag.search(query_embedding, k=5)
    这个模块实现了检索增强生成（Retrieval Augmented Generation，RAG）技术在时间序列数据处理中的应用，主要功能包括：
    将时间序列数据转换为嵌入向量（embedding）
    存储这些嵌入向量和原始数据
    提供基于向量相似性的搜索功能，用于查找相似的时间序列模式
"""

import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union


@dataclass
## 每个轨迹文档
class TimeSeriesDocument:
    """A dataclass representing a time series document with metadata and embedding.

    This class stores all information related to a time series, including its raw
    data, metadata, and vector embedding for similarity search.

    Attributes:
        id (str): Unique identifier for the time series.
        data (np.ndarray): Raw time series data.
        metadata (Dict[str, Any]): Additional information about the time series.
        embedding (Optional[np.ndarray]): Vector embedding of the time series,
            used for similarity search. Default is None.

    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        >>> doc = TimeSeriesDocument(
        ...     id="example_1",
        ...     data=data,
        ...     metadata={"type": "example"},
        ...     embedding=np.array([0.1, 0.2, 0.3])
        ... )
        id：轨迹标识符。

        data：原始轨迹数据（历史轨迹）。

        pred_data：预测结果或未来轨迹（在 search_batch 中被返回，暗示了 RAG 的应用）。

        embedding：用于相似性搜索的向量。
        这个文档描述了TimeSeriesDocument数据类的核心功能和结构，该类主要用于：
        数据组织：将时间序列数据与其相关信息组织成一个统一的结构
        相似性搜索支持：通过存储向量嵌入（embedding），为相似时间序列的快速检索提供基础
        元数据管理：允许存储额外的描述性信息，如数据类型、来源等
        在RAG（检索增强生成）系统中，此类通常作为基本的数据单元，用于构建时间序列知识库，支持后续的相似模式检索和利用。
    """

    id: str
    data: np.ndarray
    pred_data: np.ndarray
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

# 初始化向量数据库和存储时间序列“文档”
class TimeSeriesRAG:
    """A class implementing Retrieval Augmented Generation for time series data.

    This class provides functionality for storing time series documents and
    retrieving similar patterns using FAISS vector similarity search.

    Attributes:
        embedding_dim (int): Dimension of the time series embeddings.
        index (faiss.Index): FAISS index for similarity search.
        documents (List[TimeSeriesDocument]): List of stored time series documents.

    Example:
        # >>> rag = TimeSeriesRAG(embedding_dim=260)
        # >>> doc = TimeSeriesDocument(...)
        # >>> rag.add_document(doc)
        # >>> results = rag.search(query_embedding, k=5)
    """


    def __init__(self, embedding_dim: int = 780):
        """Initialize the TimeSeriesRAG system.

        Args:
            embedding_dim (int, optional): Dimension of the time series embeddings.
                Should match the output dimension of your embedding model.
                Defaults to 260 (256 resampled points + 4 statistical features).
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        # self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[TimeSeriesDocument] = []

    def add_document(self, doc: TimeSeriesDocument) -> None:
        """Add a time series document to the RAG system.

        Args:
            doc (TimeSeriesDocument): The document to add. Must have a valid
                embedding for similarity search.
        Raises:
            ValueError: If the document's embedding is None or has incorrect shape.
        """
        if doc.embedding is None:
            raise ValueError("Document must have an embedding")

        if doc.embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {doc.embedding.shape[-1]}"
            )
        doc.embedding = doc.embedding.astype(np.float32)
        self.index.add(doc.embedding.reshape(1, -1))
        self.documents.append(doc)

    # 搜索相似时间序列模式的核心检索方法。
    # 该函数作为RAG（检索增强生成）系统的关键组件，
    # 通过向量相似度匹配找到与查询序列最相似的历史时间序列文档

    def search(
            self,
            query_embedding: np.ndarray,
            k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for similar time series patterns.

        Args:
            query_embedding (np.ndarray): The embedding vector of the query time
                series. Must match the embedding dimension of the index.
            k (int, optional): Number of nearest neighbors to retrieve.
                Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search results.
                Each dictionary has the following keys:
                - 'id': Document ID
                - 'distance': L2 distance to query
                - 'data': Raw time series data
                - 'metadata': Document metadata

        Raises:
            ValueError: If query_embedding has incorrect shape.
        """
        if query_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {self.embedding_dim}, "
                f"got {query_embedding.shape[-1]}"
            )

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'id': doc.id,
                    'distance': float(distances[0][i]),
                    'data': doc.data.tolist(),
                    'metadata': doc.metadata
                })
        return results
    # 批量搜索相似时间序列模式
    def search_batch(
            self,
            query_embedding: np.ndarray,
            k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for similar time series patterns.

        Args:
            query_embedding (np.ndarray): The embedding vector of the query time
                series. Must match the embedding dimension of the index.
            k (int, optional): Number of nearest neighbors to retrieve.
                Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing search results.
                Each dictionary has the following keys:
                - 'id': Document ID
                - 'distance': L2 distance to query
                - 'data': Raw time series data
                - 'metadata': Document metadata

        Raises:
            ValueError: If query_embedding has incorrect shape.
        """
        ## 批量搜索相似时间序列模式
        batch_results = []
        for n in range(query_embedding.shape[0]):
            new_query_embedding = query_embedding[n]
            if new_query_embedding.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch. Expected {self.embedding_dim}, "
                    f"got {new_query_embedding.shape[-1]}"
                )

            distances, indices = self.index.search(
                new_query_embedding.reshape(1, -1), k
            )

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'id': doc.id,
                        'distance': float(distances[0][i]),
                        'data': doc.data.tolist(),
                        'metadata': doc.metadata,
                        'pred_data': doc.pred_data,
                    })
            batch_results.append(results)
        return batch_results

    def get_document_by_id(self, doc_id: str) -> Optional[TimeSeriesDocument]:
        """Retrieve a document by its ID.

        Args:
            doc_id (str): The ID of the document to retrieve.

        Returns:
            Optional[TimeSeriesDocument]: The document if found, None otherwise.
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None