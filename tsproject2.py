import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정 및 스타일 초기화
st.set_page_config(
    page_title="다변량 시계열 자동 이상탐지 솔루션",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태(Session State) 초기화 - 사용자의 현장 메모를 저장하는 공간
if 'memo_storage' not in st.session_state:
    st.session_state.memo_storage = {}

st.title("📊 다변량 시계열 자동 이상탐지 플랫폼")
st.markdown("임의의 다변량 시계열 CSV 파일을 업로드하여 자동으로 상관관계를 분석하고 이상치를 탐지합니다.")

# 2. 사이드바 - 데이터 업로드 및 하이퍼파라미터 설정
st.sidebar.header("📁 데이터 업로드 및 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 인공지능 모델 설정")
contamination_rate = st.sidebar.slider(
    "예상 이상치 비율 (Contamination)", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01
)
random_state = st.sidebar.number_input("Random State", value=42)

# 3. 메인 화면 로직
if uploaded_file is not None:
    # 데이터 로드 및 캐싱
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    try:
        df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # 데이터 기본 정보 파악
    columns = df_raw.columns.tolist()
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # 컬럼 선택 UI
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        time_col = st.selectbox("📅 시간(Time/Date) 기준 컬럼 선택", columns)
    with col_sel2:
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        feature_cols = st.multiselect("🧬 분석할 다변량 변수 선택 (복수 선택 가능)", numeric_cols, default=numeric_cols[:3])

    if not feature_cols:
        st.warning("⚠️ 이상탐지를 수행할 변수를 최소 1개 이상 선택해주세요.")
        st.stop()

    # 데이터 정렬 및 전처리
    df = df_raw.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # -------------------------------------------------------------
    # 차별화 UX 1: 차별화된 데이터 관계성 분석 (상관관계 Heatmap)
    # -------------------------------------------------------------
    with st.expander("🔗 차별화 분석: 다변량 변수 간 상관관계 열지도 (Heatmap)", expanded=True):
        st.markdown("##### 🌡️ 피어슨 상관계수 행렬 (Pearson Correlation Matrix)")
        
        corr_matrix = df[feature_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=320,
            template="plotly_white"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("💡 변수 간 상관관계를 사전에 파악하여, 다변량 시계열 패턴이 정상 궤적을 벗어나는지 확인하는 기초 지표로 삼습니다.")

    # 4. 자동 이상탐지 모델 수행 (Isolation Forest)
    with st.spinner("🤖 다변량 패턴 학습 및 알고리즘 연산 중..."):
        X = df[feature_cols].interpolate(method='linear').bfill().ffill()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = IsolationForest(contamination=contamination_rate, random_state=random_state)
        df['Anomaly_Code'] = model.fit_predict(X_scaled)
        df['Is_Anomaly'] = df['Anomaly_Code'].apply(lambda x: '이상' if x == -1 else '정상')
        df['Anomaly_Score'] = model.score_samples(X_scaled)

    # -------------------------------------------------------------
    # 차별화 UX 2: 시스템 실시간 위험도 신호등 (Health Status Card)
    # -------------------------------------------------------------
    total_points = len(df)
    anomaly_points = len(df[df['Anomaly_Code'] == -1])
    actual_anomaly_rate = (anomaly_points / total_points) * 100

    if actual_anomaly_rate > 10:
        status_color = "#FF4B4B"  # Soft Red
        status_text = "🚨 위험 (Critical Status)"
        status_desc = "이상치 발생 비율이 임계치(10%)를 초과했습니다. 즉각적인 시스템 점검이 필요합니다."
    elif actual_anomaly_rate > 5:
        status_color = "#FFA500"  # Orange
        status_text = "⚠️ 주의 (Warning Status)"
        status_desc = "잠재적 이상 징후가 포착되었습니다. 운영 데이터 추이를 모니터링하세요."
    else:
        status_color = "#2EB872"  # Green
        status_text = "✅ 정상 (Stable Status)"
        status_desc = "시스템이 안정적인 다변량 동적 균형 상태를 유지하고 있습니다."

    st.markdown(
        f"""
        <div style="background-color: {status_color}; padding: 20px; border-radius: 10px; color: white; margin-top: 15px; margin-bottom: 25px;">
            <h3 style="margin: 0; color: white;">{status_text}</h3>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 15px;">{status_desc} (현재 탐지된 이상 비율: <b>{actual_anomaly_rate:.2f}%</b>)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. 대시보드 메트릭 및 시각화 구성
    st.subheader("📈 이상탐지 종합 통계 및 통합 대시보드")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 데이터 수", f"{total_points:,} 수")
    m2.metric("탐지된 이상치", f"{anomaly_points:,} 수", delta=f"{actual_anomaly_rate:.2f}%", delta_color="inverse")
    m3.metric("정상 데이터", f"{total_points - anomaly_points:,} 수")
    m4.metric("평균 이상 위험 점수", f"{df['Anomaly_Score'].mean():.3f}")

    # 좌우 화면 분할 레이아웃
    chart_col, stat_col = st.columns([3, 1])

    with chart_col:
        selected_viz_col = st.selectbox("🔍 모니터링할 핵심 변수 선택", feature_cols)
        
        # 인터랙티브 시계열 차트 구현
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[selected_viz_col],
            mode='lines', name='정상 데이터 수조', line=dict(color='#A2D2FF', width=1.5)
        ))
        
        anomalies = df[df['Anomaly_Code'] == -1]
        fig.add_trace(go.Scatter(
            x=anomalies[time_col], y=anomalies[selected_viz_col],
            mode='markers', name='❌ 탐지된 이상점',
            marker=dict(color='#FF4B4B', size=7, symbol='circle')
        ))
        
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=15, r=15, t=15, b=15),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with stat_col:
        st.markdown("##### 🔲 데이터 그룹별 평균 분포")
        summary_stats = df.groupby('Is_Anomaly')[feature_cols].mean().T
        st.dataframe(summary_stats, use_container_width=True)
        
        # 스코어 히스토그램
        fig_hist = px.histogram(
            df, x='Anomaly_Score', color='Is_Anomaly',
            color_discrete_map={'정상': '#4A90E2', '이상': '#FF4B4B'},
            labels={'Anomaly_Score': '위험도 점수'}
        )
        fig_hist.update_layout(
            showlegend=False, 
            margin=dict(l=10, r=10, t=10, b=10), 
            height=200, 
            template='plotly_white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------
    # 차별화 UX 3: 도메인 전문가 피드백 및 현장 메모 작성 기능
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📝 차별화 기능: 도메인 전문가 이상 원인 분석 및 라벨링")
    st.caption("AI 모델이 탐지한 이상 시점을 선택하여 현장 기록을 남기고 데이터셋을 고도화할 수 있습니다.")
    
    anomaly_df = df[df['Anomaly_Code'] == -1].copy()
    
    if not anomaly_df.empty:
        # 시간 컬럼을 스트링으로 변환하여 선택 박스 생성
        anomaly_df['time_str'] = anomaly_df[time_col].astype(str)
        
        c_memo1, c_memo2, c_memo3 = st.columns([1.5, 2, 1])
        
        with c_memo1:
            selected_time_str = st.selectbox("📍 분석할 이상치 발생 시점", anomaly_df['time_str'].tolist())
        with c_memo2:
            # 기존 기록이 세션에 있다면 불러오기 기본값 지정
            exist_memo = st.session_state.memo_storage.get(selected_time_str, {}).get('memo', '')
            user_memo = st.text_input(f"해당 시점({selected_time_str[-8:]})의 원인 기록", value=exist_memo, placeholder="예: 센서 노이즈 감지, 장비 일시 전원 오프 등")
        with c_memo3:
            exist_cat = st.session_state.memo_storage.get(selected_time_str, {}).get('category', '기계 결함')
            memo_cat = st.selectbox("분류 태그", ["기계 결함", "센서 노이즈", "정상 노이즈(오탐)", "휴먼 에러"], index=["기계 결함", "센서 노이즈", "정상 노이즈(오탐)", "휴먼 에러"].index(exist_cat))
            
        # [저장하기] 버튼 클릭 시 세션 상태에 저장
        if st.button("💾 현장 분석 기록 저장"):
            st.session_state.memo_storage[selected_time_str] = {
                'memo': user_memo,
                'category': memo_cat
            }
            st.success("✅ 메모가 내부 데이터에 정상 반영되었습니다.")

        # 메모 결합된 최종 이상치 리포트 테이블 생성
        anomaly_df['현장 기록 메모'] = anomaly_df['time_str'].map(lambda x: st.session_state.memo_storage.get(x, {}).get('memo', '-'))
        anomaly_df['이상 분류 태그'] = anomaly_df['time_str'].map(lambda x: st.session_state.memo_storage.get(x, {}).get('category', '-'))
        
        # 테이블 뷰 출력
        st.markdown("##### 🔍 분석 메모가 포함된 이상치 통합 리포트")
        display_cols = [time_col] + feature_cols + ['Anomaly_Score', '이상 분류 태그', '현장 기록 메모']
        st.dataframe(anomaly_df[display_cols].sort_values(by=time_col), use_container_width=True)
        
        # 다운로드 기능
        final_csv = anomaly_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 전문가 분석 메모가 포함된 CSV 리포트 다운로드",
            data=final_csv,
            file_name="anomaly_expert_report.csv",
            mime="text/csv"
        )
    else:
        st.info("현재 모델 설정으로는 탐지된 이상치가 없어 메모 기능을 비활성화합니다.")

else:
    # 📌 파일 업로드 전 웰컴 대시보드 화면 (UX 향상)
    st.markdown(
        """
        <div style="border: 2px dashed #4A90E2; padding: 50px; border-radius: 12px; text-align: center; background-color: #F8FAFC; margin-top: 30px;">
            <h2 style="color: #4A90E2; margin-bottom: 15px;">📁 분석할 다변량 시계열 CSV 파일을 업로드해 주세요</h2>
            <p style="color: #64748B; max-width: 600px; margin: 0 auto 20px auto; font-size: 15px;">
                왼쪽 사이드바의 업로더를 통해 데이터를 제출하시면 자동으로 다변량 상관분석 및 Isolation Forest 기반의 시중 이상 탐지와 전문가 피드백 대시보드가 생성됩니다.
            </p>
            <span style="font-size: 13px; color: #94A3B8;">※ 필수 규칙: 시간 축을 나타내는 컬럼과 1개 이상의 수치형 데이터 변수가 포함되어야 합니다.</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
