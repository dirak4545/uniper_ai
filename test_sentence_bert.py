import chromadb
from chromadb.utils import embedding_functions

# sentence-BERT 임베딩 함수 테스트
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # sentence-BERT 임베딩 함수 생성
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 새로운 컬렉션 생성 (sentence-BERT 사용)
    collection = client.get_or_create_collection(
        name="test_sbert_collection",
        embedding_function=sentence_transformer_ef
    )
    
    # 한국어 문서 추가
    documents = [
        "대학생활에서 가장 중요한 것은 학습과 인간관계입니다.",
        "시험 준비는 계획적으로 하는 것이 좋습니다.",
        "동아리 활동을 통해 다양한 경험을 쌓을 수 있습니다.",
        "졸업 후 진로에 대해 미리 고민하고 준비해야 합니다."
    ]
    
    collection.add(
        documents=documents,
        ids=["doc1", "doc2", "doc3", "doc4"]
    )
    
    # 한국어 검색 테스트
    results = collection.query(
        query_texts=["학업 관련 조언"],
        n_results=2
    )
    
    print("sentence-BERT 임베딩 연동 성공!")
    print("검색 결과:")
    print(f"검색어: 학업 관련 조언")
    print(f"결과: {results['documents'][0]}")
    print(f"거리값: {results['distances'][0]}")
    
except Exception as e:
    print(f"sentence-BERT 연동 실패: {e}")