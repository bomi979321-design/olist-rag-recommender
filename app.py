"""
app.py
RAG 기반 Olist 상품 추천 Streamlit 앱
- faiss_index.bin / faiss_metadata.pkl 없으면 자동 빌드
- 사이드바 필터 (카테고리 / 가격 / 평점 / 결과 수)
- 의미 검색 + 결과 카드 UI
"""

import sys
import math
import logging
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# 페이지 설정 (반드시 최상단)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DATA_PATH        = Path("data/enriched_products_final.csv")
FAISS_INDEX_PATH = Path("faiss_index.bin")
FAISS_META_PATH  = Path("faiss_metadata.pkl")
MODEL_NAME       = "all-MiniLM-L6-v2"
BATCH_SIZE       = 500

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 캐시: CSV 로드
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# 캐시: 임베딩 모델 로드
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# 벡터 DB 빌드 (faiss_index.bin 없을 때만)
# ---------------------------------------------------------------------------
def build_vectorstore(df: pd.DataFrame, model) -> None:
    import faiss
    import pickle
    import numpy as np

    st.info("벡터 DB를 처음 구축합니다. 약 20~30분 소요됩니다.")
    progress = st.progress(0, text="임베딩 생성 중...")

    # 임베딩 텍스트 구성
    def make_text(row):
        reviews = row["aggregated_reviews"]
        if pd.isna(reviews) or str(reviews).strip() == "":
            reviews = "No reviews available"
        rating  = f"{row['avg_review_score']:.1f}" if pd.notna(row["avg_review_score"]) else "N/A"
        price   = f"{row['avg_price']:.2f}"        if pd.notna(row["avg_price"])        else "N/A"
        return (
            f"Product: {row['product_name']}\n"
            f"Category: {row['product_category_name_english']}\n"
            f"Price: ${price}\n"
            f"Rating: {rating}/5.0\n"
            f"Description: {row['description']}\n"
            f"Reviews: {reviews}"
        )

    documents = df.apply(make_text, axis=1).tolist()
    metadatas = [
        {
            "product_id":   str(r["product_id"]),
            "product_name": str(r["product_name"]),
            "category":     str(r["product_category_name_english"]) if pd.notna(r["product_category_name_english"]) else "",
            "price":        float(r["avg_price"])        if pd.notna(r["avg_price"])        else 0.0,
            "review_score": float(r["avg_review_score"]) if pd.notna(r["avg_review_score"]) else 0.0,
        }
        for _, r in df.iterrows()
    ]

    # 배치 임베딩
    n = len(documents)
    n_batches = math.ceil(n / BATCH_SIZE)
    all_vecs = []
    for i in range(n_batches):
        batch = documents[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        vecs  = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_vecs.append(vecs)
        progress.progress((i + 1) / n_batches, text=f"임베딩 생성 중... {min((i+1)*BATCH_SIZE, n):,}/{n:,}")

    # FAISS 인덱스 구축 (코사인 유사도 = 정규화 벡터의 내적)
    progress.progress(1.0, text="FAISS 인덱스 저장 중...")
    matrix = np.vstack(all_vecs).astype("float32")
    dim    = matrix.shape[1]
    index  = faiss.IndexFlatIP(dim)   # Inner Product on normalized vectors = cosine
    index.add(matrix)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(metadatas, f)

    progress.empty()
    st.success(f"벡터 DB 구축 완료! {n:,}개 상품이 저장되었습니다.")


# ---------------------------------------------------------------------------
# 캐시: FAISS 인덱스 + 메타데이터 로드
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_faiss_store():
    import faiss
    import pickle
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "rb") as f:
        metadatas = pickle.load(f)
    return index, metadatas


# ---------------------------------------------------------------------------
# 검색 함수
# ---------------------------------------------------------------------------
def search(query: str, model, faiss_store, top_k: int = 20):
    import numpy as np
    index, metadatas = faiss_store
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(vec, top_k)
    items = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(metadatas[idx])
        item["similarity"] = round(float(score), 4)
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# UI 헬퍼: 결과 카드
# ---------------------------------------------------------------------------
def render_card(item: dict, rank: int) -> None:
    sim   = item["similarity"]
    price = item.get("price", 0.0)
    score = item.get("review_score", 0.0)
    cat   = item.get("category", "").replace("_", " ").title()
    name  = item.get("product_name", "Unknown")

    # 제품명 + 메타 정보
    st.markdown(f"**{rank}. {name}**")
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.caption(f"📂 {cat}")
    col2.caption(f"💰 ${price:.2f}")
    col3.caption(f"⭐ {score:.1f} / 5.0")

    # 유사도 progress bar
    pct = int(sim * 100)
    bar_color = "#1D9E75" if sim >= 0.55 else "#f0a500" if sim >= 0.40 else "#aaa"
    st.markdown(
        f"""<div style='margin:4px 0 8px 0'>
        <span style='font-size:0.75rem;color:#888'>유사도 {pct}%</span>
        <div style='height:6px;background:#eee;border-radius:3px;margin-top:3px'>
          <div style='width:{pct}%;height:6px;background:{bar_color};border-radius:3px'></div>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # description (collection에 저장 안 됨 → CSV에서 가져옴)
    product_id = item.get("product_id", "")
    if "df_cache" in st.session_state and product_id:
        row = st.session_state["df_cache"].loc[
            st.session_state["df_cache"]["product_id"] == product_id
        ]
        if not row.empty:
            desc = row.iloc[0].get("description", "")
            if desc:
                st.caption(desc)

    st.divider()


# ---------------------------------------------------------------------------
# 메인 앱
# ---------------------------------------------------------------------------
def main():
    # ── 사이드바 ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🛍️ Olist Product\nRecommender")
        st.markdown("---")

        # 데이터 로드 (사이드바 렌더링 전에 필요)
        if not DATA_PATH.exists():
            st.error(f"데이터 파일을 찾을 수 없습니다.\n\n`{DATA_PATH}` 경로에 CSV 파일을 배치해 주세요.")
            st.stop()

        with st.spinner("데이터 로드 중..."):
            df = load_csv()
        st.session_state["df_cache"] = df

        all_cats = sorted(df["product_category_name_english"].dropna().unique().tolist())
        display_cats = [c.replace("_", " ").title() for c in all_cats]
        cat_map = dict(zip(display_cats, all_cats))  # display -> raw

        st.subheader("필터")
        selected_display = st.multiselect(
            "카테고리",
            options=display_cats,
            default=[],
            placeholder="전체 카테고리",
        )
        selected_cats = [cat_map[d] for d in selected_display]

        price_range = st.slider(
            "가격 범위 ($)",
            min_value=0,
            max_value=500,
            value=(0, 500),
            step=10,
        )

        min_score = st.slider(
            "최소 평점",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
        )

        top_k_choice = st.selectbox(
            "검색 결과 수",
            options=[3, 5, 10],
            index=1,
        )

        st.markdown("---")
        st.caption("powered by FAISS + sentence-transformers")

    # ── 벡터 DB 초기화 ────────────────────────────────────────────────────
    with st.spinner("모델 로드 중..."):
        model = load_model()

    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        build_vectorstore(df, model)
        st.cache_resource.clear()   # 새로 만든 인덱스 캐시 반영
        st.rerun()

    try:
        faiss_store = load_faiss_store()
    except Exception as e:
        st.error(f"FAISS 인덱스 로드 실패: {e}\n\nfaiss_index.bin / faiss_metadata.pkl 을 삭제하고 앱을 재시작해 주세요.")
        st.stop()

    # ── 메인 헤더 ─────────────────────────────────────────────────────────
    st.markdown("## 🔍 상품 추천 검색")
    st.markdown("자연어로 원하는 상품을 검색하세요.")

    # ── 세션 상태 초기화 ──────────────────────────────────────────────────
    if "query"   not in st.session_state: st.session_state["query"]   = ""
    if "history" not in st.session_state: st.session_state["history"] = []

    # ── 인기 카테고리 퀵버튼 ──────────────────────────────────────────────
    if not st.session_state["query"]:
        st.markdown("##### 인기 카테고리 바로가기")
        quick_cols = st.columns(5)
        quick_items = [
            ("💻 Electronics",  "wireless electronics"),
            ("🧸 Toys",         "children toys and games"),
            ("👗 Fashion",      "clothing and fashion accessories"),
            ("🏠 Home & Living","home furniture and decor"),
            ("💄 Beauty",       "beauty skincare perfume"),
        ]
        for col, (label, kw) in zip(quick_cols, quick_items):
            if col.button(label, use_container_width=True):
                st.session_state["query"] = kw
                st.rerun()

    # ── 검색창 ────────────────────────────────────────────────────────────
    query_input = st.text_input(
        "What are you looking for?",
        value=st.session_state["query"],
        placeholder='e.g. wireless headphone under $50',
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 5])
    search_clicked = col_btn.button("🔍 Search", type="primary", use_container_width=True)
    if col_clear.button("✕ Clear", use_container_width=False):
        st.session_state["query"] = ""
        st.rerun()

    # ── 검색 실행 ─────────────────────────────────────────────────────────
    if search_clicked and query_input.strip():
        st.session_state["query"] = query_input.strip()

        # 히스토리 저장
        hist = st.session_state["history"]
        if query_input not in hist:
            hist.insert(0, query_input)
            st.session_state["history"] = hist[:10]

        with st.spinner("검색 중..."):
            raw_results = search(query_input, model, faiss_store, top_k=50)

        # 필터 적용
        filtered = []
        for item in raw_results:
            if selected_cats and item.get("category", "") not in selected_cats:
                continue
            price = item.get("price", 0.0)
            if not (price_range[0] <= price <= price_range[1]):
                continue
            if item.get("review_score", 0.0) < min_score:
                continue
            filtered.append(item)
            if len(filtered) >= top_k_choice:
                break

        # 결과 출력
        if not filtered:
            st.warning("No products found. Try different keywords or adjust filters.")
        else:
            st.markdown(f"**{len(filtered)}개** 결과 (쿼리: *{query_input}*)")
            for rank, item in enumerate(filtered, start=1):
                render_card(item, rank)

    elif not search_clicked and st.session_state["query"]:
        # 검색어는 있지만 버튼 안 누른 상태 → 이전 결과 유지 안 함 (깔끔한 UX)
        pass

    # ── 검색 히스토리 ─────────────────────────────────────────────────────
    if st.session_state["history"]:
        with st.expander("🕐 최근 검색어", expanded=False):
            for h in st.session_state["history"]:
                if st.button(f"↩ {h}", key=f"hist_{h}"):
                    st.session_state["query"] = h
                    st.rerun()


if __name__ == "__main__":
    main()
