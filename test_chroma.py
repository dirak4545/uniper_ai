import chromadb

# ChromaDB 연결 테스트
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("test_collection")
    
    # 문서 추가
    collection.add(
        documents=["이것은 테스트 문서입니다.", "ChromaDB 연동이 성공했습니다."],
        ids=["doc1", "doc2"]
    )
    
    # 검색 테스트
    results = collection.query(
        query_texts=["테스트"],
        n_results=2
    )
    
    print("ChromaDB 연동 성공!")
    print("검색 결과:", results)
    
except Exception as e:
    print(f"ChromaDB 연동 실패: {e}")