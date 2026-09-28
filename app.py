
import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 기본 설정을 최상단에 구성 (타이틀, 레이아웃, 아이콘)
st.set_page_config(
    page_title="제조기업 관리자 3분 데이터 분석 대시보드",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_sample_manufacturing_data() -> pd.DataFrame:
    """
    업로드된 파일이 없을 때 초보자가 대시보드 기능을 바로 체험해볼 수 있는 
    자동차 부품 제조기업의 샘플 데이터셋입니다.
    """
    data = {
        "지표/데이터명": [
            "국내 자동차 생산량", "자동차 수출량", "부품 무역수지", 
            "알루미늄 LME 단가", "철광석 CFR 단가", "원/달러 환율", 
            "친환경차(HEV) 판매량", "전기차(EV) 판매량", "주요 OEM 납품비중"
        ],
        "업무분류": [
            "생산량", "생산량", "수출입", 
            "원자재", "원자재", "환율", 
            "친환경차", "친환경차", "완성차"
        ],
        "최신수치": [2058400, 1385200, 8570000000, 3242, 108.5, 1358.5, 298000, 118000, 85.4],
        "전월수치": [1980000, 1310000, 8100000000, 2980, 110.0, 1348.0, 250000, 125000, 84.0],
        "단위": ["대", "대", "USD", "USD/톤", "USD/톤", "원", "대", "대", "%"],
        "위험도": ["NORMAL", "NORMAL", "NORMAL", "HIGH", "NORMAL", "HIGH", "HIGH", "NORMAL", "NORMAL"],
        "담당부서": ["생산기획팀", "해외영업팀", "무역관리팀", "구매팀", "구매팀", "재무회계팀", "생산기술팀", "생산기술팀", "고객지원팀"],
        "기준일자": ["2026-08-31", "2026-08-31", "2026-08-31", "2026-09-28", "2026-09-28", "2026-09-28", "2026-09-28", "2026-09-28", "2026-09-28"]
    }
    return pd.DataFrame(data)


def analyze_data_structure(df: pd.DataFrame) -> dict:
    """
    데이터프레임의 구조, 수치형/범주형 컬럼, 결측치, 이상치, 긴급 지표를 분석하는 순수 데이터 처리 함수입니다.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_sum = int(df.isnull().sum().sum())
    
    # 1. 범주형 및 수치형 컬럼 자동 판별
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # 2. 긴급/위험 항목 자동 탐지
    high_risk_rows = pd.DataFrame()
    
    # '위험도' 컬럼이 있는 경우 HIGH 항목 추출
    if "위험도" in df.columns:
        high_risk_rows = df[df["위험도"].astype(str).str.upper().str.contains("HIGH|URGENT|CRITICAL|WARNING|경보|긴급")]
    
    # 3. 수치형 컬럼 기준 이상치(IQR 기법) 탐지
    outlier_summary = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 4:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                if not outliers.empty:
                    outlier_summary.append({
                        "column": col,
                        "outlier_count": len(outliers),
                        "max_value": series.max(),
                        "min_value": series.min()
                    })

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_sum": missing_sum,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "high_risk_rows": high_risk_rows,
        "outlier_summary": outlier_summary
    }


def generate_ai_insight(df_summary: dict, df_sample_text: str, api_key: str = None) -> str:
    """
    독립된 AI 분석 함수. API Key가 설정되어 있으면 OpenAI API를 호출하고,
    설정되지 않았거나 오류 시 규칙 기반(Rule-based) 임원용 AI 종합 리포트를 자동 생성합니다.
    """
    # 1. API Key가 제공된 경우 외부 AI API 연동 시도 (OpenAI 기준 예시)
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""
            당신은 자동차 부품 제조기업의 최고운영책임자(COO)를 보좌하는 AI 전략 컨설턴트입니다.
            다음 제조/마켓 데이터 요약 정보를 보고 관리자가 3분 이내에 파악할 수 있는 Executive Summary를 작성하세요.

            [데이터 통계 요약]
            - 전체 데이터 행: {df_summary['total_rows']}건, 열: {df_summary['total_cols']}개
            - 수치형 지표: {', '.join(df_summary['num_cols'])}
            - 긴급 주의 항목 수: {len(df_summary['high_risk_rows'])}건
            
            [주요 데이터 샘플]
            {df_sample_text}

            [작성 양식]
            1. 🚨 긴급 대응 및 위험 요인 (원가/생산/환율 등)
            2. 📊 핵심 운영 KPI 및 주요 트렌드 요약
            3. 💡 관리자 권장 액션 플랜 (3가지)
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"⚠️ AI API 호출 중 오류가 발생하여 내장 AI 로직으로 대체합니다: {e}")

    # 2. API Key가 없거나 API 실패 시 작동하는 내장 제조 특화 규칙 기반 브리핑
    briefing = []
    briefing.append("### 🤖 [AI Executive Dashboard Summary]")
    briefing.append("본 리포트는 수집된 제조/원가/시장 데이터를 종합 진단한 관리자용 브리핑입니다.\n")
    
    # 🚨 긴급 주의 진단
    briefing.append("#### 1. 🚨 긴급 위험 요소 및 이상징후")
    if not df_summary['high_risk_rows'].empty:
        briefing.append(f"- **긴급 경보 항목 {len(df_summary['high_risk_rows'])}건이 감지되었습니다.**")
        for idx, row in df_summary['high_risk_rows'].iterrows():
            name = row.get("지표/데이터명", row.get(df_summary['cat_cols'][0] if df_summary['cat_cols'] else "항목", f"행 {idx}"))
            val = row.get("최신수치", row.get("val", "-"))
            unit = row.get("단위", "")
            briefing.append(f"  • **{name}**: 현재 수치 `{val} {unit}` -> 원가 상승 또는 수급 리스크 점검 필요")
    else:
        briefing.append("- 특이 긴급 경보 항목은 발견되지 않았으며, 정상 관리 범위 내에 있습니다.")
        
    if df_summary['outlier_summary']:
        briefing.append("- **수치 이상치(Outlier) 감지:**")
        for out in df_summary['outlier_summary']:
            briefing.append(f"  • `{out['column']}` 컬럼에서 표준 범위를 벗어난 수치 {out['outlier_count']}건 탐지 (최대값: {out['max_value']})")

    # 📊 KPI 분석
    briefing.append("\n#### 2. 📊 핵심 운영 지표 트렌드")
    briefing.append(f"- 총 **{df_summary['total_rows']}개 지표** 모니터링 중 (수치 컬럼 {len(df_summary['num_cols'])}개, 범주 컬럼 {len(df_summary['cat_cols'])}개)")
    briefing.append(f"- 결측치(Data Quality): 총 {df_summary['missing_sum']}건으로 데이터 무결성 상태 점검 완료.")

    # 💡 Action Plan
    briefing.append("\n#### 3. 💡 관리자 즉시 실행 권장안 (Action Plan)")
    briefing.append("1. **원가 연동제 단가 재협상**: 알루미늄/원자재 가격 급등 항목에 대해 완성차 구매실과 납품 단가 인상 건 발의")
    briefing.append("2. **생산 믹스 가동률 조정**: 하이브리드(HEV) 등 수요가 급증하는 부품 라인의 우선 배정 및 가동률 상향")
    briefing.append("3. **외환 결제 헤지 점검**: 고환율 유지 구간에서의 달러/엔화 결제 시점 분산 조정")

    return "\n".join(briefing)


# --- 상단 헤더 영역 ---
st.title("🏭 제조기업 관리자용 3분 상황파악 대시보드")
st.caption("자동차 부품 및 제조 현장의 핵심 KPI, 이상징후 감지, AI 리포트 종합 분석 시스템")

# --- 사이드바 설정 영역 ---
st.sidebar.header("⚙️ 대시보드 컨트롤 패널")

# 1. 파일 업로더 (CSV & XLSX 지원)
uploaded_file = st.sidebar.file_uploader(
    "📁 CSV 또는 Excel 파일 업로드", 
    type=["csv", "xlsx", "xls"],
    help="제조, 원가, 생산 관련 CSV 또는 XLSX 파일을 드래그하여 업로드하세요."
)

# 2. API Key 설정 (Streamlit Secrets / 환경변수 / 직접입력 지원)
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 AI API Key 설정 (선택사항)")

# 환경변수 또는 streamlit secrets에서 키 가져오기 시도
env_api_key = os.getenv("OPENAI_API_KEY")
try:
    if not env_api_key and "OPENAI_API_KEY" in st.secrets:
        env_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

user_api_key = st.sidebar.text_input(
    "OpenAI API Key 입력", 
    value=env_api_key if env_api_key else "",
    type="password",
    placeholder="sk-...",
    help="API Key가 없어도 내장 규칙기능으로 AI 리포트가 생성됩니다."
)


if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
        st.sidebar.success(f"✅ 파일 업로드 성공: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다: {e}")
        df_raw = get_sample_manufacturing_data()
else:
    st.sidebar.info("💡 샘플 데이터셋으로 실행 중입니다. (파일을 업로드하여 변경 가능)")
    df_raw = get_sample_manufacturing_data()

# 데이터 기본 구조 분석 실행
analysis_res = analyze_data_structure(df_raw)


st.sidebar.markdown("---")
st.sidebar.subheader("🔍 실시간 데이터 필터")

filtered_df = df_raw.copy()

# 범주형 컬럼이 있을 경우 필터 옵션 생성
if analysis_res["cat_cols"]:
    filter_col = st.sidebar.selectbox("필터 기준 컬럼 선택", options=["전체 보기"] + analysis_res["cat_cols"])
    if filter_col != "전체 보기":
        unique_vals = df_raw[filter_col].dropna().unique().tolist()
        selected_vals = st.sidebar.multiselect(f"[{filter_col}] 값 선택", options=unique_vals, default=unique_vals)
        if selected_vals:
            filtered_df = filtered_df[filtered_df[filter_col].isin(selected_vals)]



# ---------------------------------------------------------
# [영역 1] 관리자 3분 상황파악 - 이상징후 및 긴급 항목
# ---------------------------------------------------------
st.markdown("### 🚨 관리자 긴급 확인 & 이상징후 (Urgent Executive Alerts)")

alert_col1, alert_col2 = st.columns([2, 1])

with alert_col1:
    if not analysis_res["high_risk_rows"].empty:
        st.error(f"⚠️ **[긴급 경보 발생]** 총 **{len(analysis_res['high_risk_rows'])}건**의 우려 지표가 감지되었습니다.")
        st.dataframe(
            analysis_res["high_risk_rows"], 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("🟢 **[상태 양호]** 긴급 대응이 필요한 위협 지표가 없습니다. 모든 프로세스가 정상입니다.")

with alert_col2:
    if analysis_res["outlier_summary"]:
        st.warning("📊 **[수치 이상치 감지]**")
        for item in analysis_res["outlier_summary"]:
            st.write(f"- **{item['column']}**: 이상치 {item['outlier_count']}개 (최대: {item['max_value']:,})")
    else:
        st.info("ℹ️ **[이상치 통계]** 수치 데이터가 표준 통계 범위 내에서 안정적으로 분포해 있습니다.")


# ---------------------------------------------------------
# [영역 2] 요약 메트릭 카드 (행/열/결측치/컬럼유형)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 데이터 구조 및 무결성 요약")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("총 데이터 행 수", f"{analysis_res['total_rows']:,} 개")
m2.metric("총 컬럼 수", f"{analysis_res['total_cols']:,} 개")
m3.metric("결측치 총합", f"{analysis_res['missing_sum']:,} 개", delta="양호" if analysis_res['missing_sum'] == 0 else "-확인필요", delta_color="normal" if analysis_res['missing_sum'] == 0 else "inverse")
m4.metric("수치형 컬럼", f"{len(analysis_res['num_cols'])} 개")
m5.metric("범주형 컬럼", f"{len(analysis_res['cat_cols'])} 개")


# ---------------------------------------------------------
# [영역 3] 데이터 미리보기 & CSV 다운로드
# ---------------------------------------------------------
st.markdown("---")
expander_title = f"👁️ 데이터 미리보기 및 조회 (필터링 적용: {len(filtered_df)} / {len(df_raw)} 건)"
with st.expander(expander_title, expanded=True):
    st.dataframe(filtered_df, use_container_width=True)
    
    # 결과 CSV 다운로드 버튼
    csv_data = filtered_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 필터링된 결과 CSV 다운로드",
        data=csv_data,
        file_name="filtered_manufacturing_data.csv",
        mime="text/csv",
        help="현재 화면에 보이는 필터링 데이터를 CSV 파일로 즉시 다운로드합니다."
    )


# ---------------------------------------------------------
# [영역 4] 사용자 선택 기반 시각화 차트 (막대그래프 & 선그래프)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 사용자 맞춤형 데이터 시각화")

if not analysis_res["num_cols"]:
    st.warning("⚠️ 시각화를 생성할 수 있는 수치형(Numeric) 컬럼이 존재하지 않습니다.")
else:
    c1, c2, c3 = st.columns([2, 2, 1])
    
    with c1:
        # X축 선택 (범주형 우선, 없으면 전체)
        x_axis_options = analysis_res["cat_cols"] if analysis_res["cat_cols"] else df_raw.columns.tolist()
        selected_x = st.selectbox("X축 컬럼 선택", options=x_axis_options, index=0)
        
    with c2:
        # Y축 선택 (수치형)
        selected_y = st.selectbox("Y축 (수치형) 컬럼 선택", options=analysis_res["num_cols"], index=0)
        
    with c3:
        # 색상 구분 컬럼 선택 (선택사항)
        color_options = ["None"] + analysis_res["cat_cols"]
        selected_color = st.selectbox("색상/그룹 컬럼 (선택)", options=color_options, index=0)
        color_val = None if selected_color == "None" else selected_color

    tab1, tab2 = st.tabs(["📊 막대 그래프 (Bar Chart)", "📈 선 그래프 (Line Chart)"])
    
    with tab1:
        st.subheader("막대 그래프")
        fig_bar = px.bar(
            filtered_df, 
            x=selected_x, 
            y=selected_y, 
            color=color_val,
            title=f"[{selected_x}] 별 [{selected_y}] 비교",
            text_auto='.2s',
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(xaxis_tickangle=-30, margin=dict(l=20, r=20, t=50, b=50))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab2:
        st.subheader("선 그래프")
        fig_line = px.line(
            filtered_df, 
            x=selected_x, 
            y=selected_y, 
            color=color_val,
            markers=True,
            title=f"[{selected_x}] 기준 [{selected_y}] 변동 추세",
            template="plotly_white"
        )
        fig_line.update_layout(xaxis_tickangle=-30, margin=dict(l=20, r=20, t=50, b=50))
        st.plotly_chart(fig_line, use_container_width=True)


# ---------------------------------------------------------
# [영역 5] AI 분석 버튼 & 임원용 AI 종합 리포트
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 AI 대시보드 종합 분석")

col_ai_btn, col_ai_info = st.columns([1, 3])

with col_ai_btn:
    run_ai = st.button("🚀 AI 분석 실행", type="primary", use_container_width=True)

with col_ai_info:
    st.caption("버튼을 누르면 데이터 구조, 위험 항목, 추세 지표를 종합 분석하여 임원 보고용 3분 리포트를 생성합니다.")

if run_ai or "ai_report_cache" in st.session_state:
    if run_ai:
        with st.spinner("AI가 데이터를 분석하여 브리핑 리포트를 작성 중입니다..."):
            # 텍스트 샘플 추출
            sample_str = filtered_df.head(10).to_string()
            
            # 독립된 AI 함수 호출 (분석 함수와 분리)
            report_res = generate_ai_insight(
                df_summary=analysis_res, 
                df_sample_text=sample_str, 
                api_key=user_api_key if user_api_key else None
            )
            st.session_state["ai_report_cache"] = report_res

    # AI 리포트 결과 출력
    st.info(st.session_state["ai_report_cache"])

# 푸터 영역
st.markdown("---")
st.caption("🔒 **보안 안내**: 로컬 환경에서 구동 시 업로드한 파일은 외부 서버에 저장되지 않습니다. | 자동차 부품 제조기업 대시보드 v1.0")
