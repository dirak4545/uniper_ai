#!/usr/bin/env python3
import chromadb
from chromadb.utils import embedding_functions

def check_chromadb():
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        collection = chroma_client.get_or_create_collection(
            name="uniper_collection_sbert",
            embedding_function=sentence_transformer_ef
        )

        # 컬렉션 정보 확인
        print("=== 컬렉션 정보 ===")
        print(f"컬렉션 이름: {collection.name}")
        print(f"총 문서 수: {collection.count()}")

        # 저장된 문서들 확인
        if collection.count() > 0:
            all_docs = collection.get()
            print("\n=== 저장된 문서들 ===")
            for i, (doc_id, document) in enumerate(zip(all_docs['ids'], all_docs['documents'])):
                print(f"{i+1}. ID: {doc_id}")
                content = document[:100] + "..." if len(document) > 100 else document
                print(f"   내용: {content}")
                print()

            # 메타데이터도 있다면 출력
            if all_docs.get('metadatas') and any(all_docs['metadatas']):
                print("=== 메타데이터 ===")
                for i, metadata in enumerate(all_docs['metadatas']):
                    if metadata:
                        print(f"{i+1}. {metadata}")

        else:
            print("\n저장된 문서가 없습니다.")
            
        # 테스트 검색 (문서가 있는 경우)
        if collection.count() > 0:
            print("\n=== 테스트 검색 ===")
            test_results = collection.query(
                query_texts=["테스트"],
                n_results=3
            )
            print(f"검색 결과: {len(test_results['ids'][0])}개 문서")
            
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    check_chromadb()