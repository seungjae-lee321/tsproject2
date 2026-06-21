import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. 페이지 기본 설정 및 스타일 초기화
st.set_page_config(
    page_title="다변량 시계열 자동 이상탐지 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 UI/UX 향상을 위한 전용 CSS 스타일 적용 (글꼴, 카드 배경, 테두리 등)
st.markdown("""
    <style>
        .main-title { font-size: 2.2rem !important; font-weight: 800 !important; color: #1E293B; margin-bottom: 5px; }
        .sub-title { font-size: 1rem !important; color: #64748B; margin-bottom: 25px; }
        .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }
        .stAlert { border-radius: 8px !important; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #0F172A; }
        .footer { text-align: center; margin-top: 50px; padding: 20px; color: #94A3B8; font-size: 0.85rem; border-top: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# 헤더 영역 UI
st.markdown('<div class="main-title">📊 다변량 시계열 자동 이상탐지 플랫폼</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">임의의 다변량 시계열 데이터를 실시간으로 전처리, 상관분석, 모델링하여 원인 변수까지 역추적하는 통합 관제 솔루션입니다.</div>', unsafe_allow_html=True)

# 2. 사이드바 - 디자인 및 설정 스킨 개선
st.sidebar.markdown("## 📁 데이터 소스 로드")
uploaded_file = st.sidebar.file_uploader("CSV 형식을 업로드하세요.", type=["csv"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("## ⚙️ 알고리즘 엔진 설정")
contamination_rate = st.sidebar.slider(
    "예상 이상치 비율 (Contamination)", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01,
    help="데이터 전체에서 이상치로 의심되는 최대 비율을 정의합니다."
)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

# 3. 메인 화면 로직
if uploaded_file is not None:
    # 데이터 로드 및 캐싱
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    try:
        df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"⚠️ 파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    # 데이터 기본 정보 파악
    columns = df_raw.columns.tolist()
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # 📌 UX 개선: 컬럼 선택 영역을 깔끔한 컨테이너(카드) 형태로 래핑
    with st.container(border=True):
        st.markdown("##### 🔍 1단계: 변수 매핑 및 차원 정의")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            time_col = st.selectbox("📅 시계열 기준(Time/Date) 컬럼 지정", columns)
        with col_sel2:
            if time_col in numeric_cols:
                numeric_cols.remove(time_col)
            feature_cols = st.multiselect("🧬 분석 대상 다변량 수치형 변수 선택", numeric_cols, default=numeric_cols[:3])

    if not feature_cols:
        st.info("💡 분석을 시작하려면 우측 상단에서 이상탐지에 사용할 다변량 변수를 1개 이상 선택해 주세요.")
        st.stop()

    # 데이터 정렬 및 전처리
    df = df_raw.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)

    # 4. 자동 이상탐지 모델 연산 (Isolation Forest)
    with st.spinner("🤖 AI가 다변량 동적 패턴을 학습하고 알고리즘을 연산 중입니다..."):
        X = df[feature_cols].interpolate(method='linear').bfill().ffill()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = IsolationForest(contamination=contamination_rate, random_state=random_state)
        df['Anomaly_Code'] = model.fit_predict(X_scaled)
        df['Is_Anomaly'] = df['Anomaly_Code'].apply(lambda x: '이상' if x == -1 else '정상')
        df['Anomaly_Score'] = model.score_samples(X_scaled)

    # -------------------------------------------------------------
    # 🚨 실시간 위험도 신호등 UI (Health Status Card)
    # -------------------------------------------------------------
    total_points = len(df)
    anomaly_points = len(df[df['Anomaly_Code'] == -1])
    actual_anomaly_rate = (anomaly_points / total_points) * 100

    if actual_anomaly_rate > 10:
        status_color = "#EF4444"; status_text = "🚨 시스템 위험 (Critical Status)"
        status_desc = "이상 발생 비율이 임계치(10%)를 초과했습니다. 즉각적인 시스템 점검 및 원인 조사가 필요합니다."
    elif actual_anomaly_rate > 5:
        status_color = "#F59E0B"; status_text = "⚠️ 시스템 주의 (Warning Status)"
        status_desc = "잠재적 불량 징후가 포착되었습니다. 운영 데이터의 실시간 추이를 면밀히 모니터링하세요."
    else:
        status_color = "#10B981"; status_text = "✅ 시스템 정상 (Stable Status)"
        status_desc = "시스템이 안정적인 다변량 동적 균형 상태를 원활히 유지하고 있습니다."

    st.markdown(
        f"""
        <div style="background-color: {status_color}; padding: 18px; border-radius: 10px; color: white; margin-top: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h4 style="margin: 0; color: white; font-weight:700;">{status_text}</h4>
            <p style="margin: 6px 0 0 0; opacity: 0.95; font-size: 0.95rem;">{status_desc} (실제 탐지된 이상 비율: <b>{actual_anomaly_rate:.2f}%</b>)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 📌 UX 개선: 탭(Tabs) 레이아웃을 도입하여 대시보드를 깔끔하게 분할
    tab1, tab2, tab3 = st.tabs(["📊 종합 관제 대시보드", "🔗 다변량 상관분석", "🕵️ 원인 변수 추적 (XAI)"])

    # -------------------------------------------------------------
    # TAB 1: 종합 관제 대시보드
    # -------------------------------------------------------------
    with tab1:
        # 주요 핵심 지표 4개 배치
        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 관측 샘플 수", f"{total_points:,} 개")
            m2.metric("탐지된 이상치 수", f"{anomaly_points:,} 개", delta=f"{actual_anomaly_rate:.2f}%", delta_color="inverse")
            m3.metric("안정 정상 샘플 수", f"{total_points - anomaly_points:,} 개")
            m4.metric("평균 위험도 지표", f"{df['Anomaly_Score'].mean():.3f}")

        # 메인 가시화 영역 분할
        chart_col, stat_col = st.columns([2.5, 1])

        with chart_col:
            selected_viz_col = st.selectbox("📈 시계열 실시간 모니터링 변수 변경", feature_cols)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[time_col], y=df[selected_viz_col],
                mode='lines', name='정상 추이', line=dict(color='#38BDF8', width=1.5)
            ))
            
            anomalies = df[df['Anomaly_Code'] == -1]
            fig.add_trace(go.Scatter(
                x=anomalies[time_col], y=anomalies[selected_viz_col],
                mode='markers', name='❌ 탐지된 이상치',
                marker=dict(color='#EF4444', size=7, symbol='circle')
            ))
            
            fig.update_layout(
                template='plotly_white', hovermode='x unified',
                margin=dict(l=10, r=10, t=10, b=10), height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        with stat_col:
            st.markdown("##### 🔲 그룹별 평균 비교")
            summary_stats = df.groupby('Is_Anomaly')[feature_cols].mean().T
            st.dataframe(summary_stats, use_container_width=True)
            
            fig_hist = px.histogram(
                df, x='Anomaly_Score', color='Is_Anomaly',
                color_discrete_map={'정상': '#3B82F6', '이상': '#EF4444'}, labels={'Anomaly_Score': '위험도 스코어'}
            )
            fig_hist.update_layout(showlegend=False, margin=dict(l=5, r=5, t=5, b=5), height=180, template='plotly_white')
            st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: 다변량 상관분석
    # -------------------------------------------------------------
    with tab2:
        st.markdown("##### 🌡️ 피어슨 상관계수 열지도 (Pearson Correlation Heatmap)")
        st.caption("변수 간 선형적 연관성을 시각화하여 어떤 변수 쌍이 강하게 묶여 움직이는지 파악합니다.")
        
        corr_matrix = df[feature_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2), texttemplate="%{text}", hoverinfo="z"
        ))
        fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: 원인 변수 추적 (XAI)
    # -------------------------------------------------------------
    with tab3:
        st.markdown("##### 🕵️ 이상 발생 핵심 원인 변수 추적기 (Root Cause Analyzer)")
        
        if not anomalies.empty:
            df_numeric = pd.DataFrame(X_scaled, columns=feature_cols)
            df_numeric[time_col] = df[time_col]
            df_numeric['Anomaly_Code'] = df['Anomaly_Code']
            
            anomaly_scaled = df_numeric[df_numeric['Anomaly_Code'] == -1].copy()
            anomaly_scaled['time_str'] = anomaly_scaled[time_col].astype(str)
            
            # 사용자 시점 선택 UI
            selected_anomaly_time = st.selectbox("📍 분석하고자 하는 이상치 발생 시점을 픽업하세요.", anomaly_scaled['time_str'].tolist())
            
            row_data = anomaly_scaled[anomaly_scaled['time_str'] == selected_anomaly_time][feature_cols].iloc[0]
            contribution = np.abs(row_data) 
            
            df_contrib = pd.DataFrame({
                '변수명': feature_cols,
                '이상 기여도 (정상 표준범위 이탈 거리)': contribution.values
            }).sort_values(by='이상 기여도 (정상 표준범위 이탈 거리)', ascending=True)
            
            fig_cause = px.bar(
                df_contrib, x='이상 기여도 (정상 표준범위 이탈 거리)', y='변수명', orientation='h',
                color='이상 기여도 (정상 표준범위 이탈 거리)', color_continuous_scale='YlOrRd'
            )
            fig_cause.update_layout(template='plotly_white', height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cause, use_container_width=True)
            
            top_variable = df_contrib.iloc[-1]['변수명']
            st.warning(f"💡 **분석 결과 진단:** 선택하신 시점에는 **[{top_variable}]** 변수가 평상시 균형(평균값)을 가장 심하게 이탈하여 전체 다변량 이상을 초래한 **지배적 원인(Root Cause)**으로 판명되었습니다.")
        else:
            st.info("현재 모델 설정 상 탐지된 이상치가 없으므로 원인 분석 요소를 생략합니다.")

    # -------------------------------------------------------------
    # 데이터 내보내기 영역
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📥 탐지된 전체 데이터셋 테이블 및 내보내기", expanded=False):
        st.dataframe(df[[time_col] + feature_cols + ['Anomaly_Score', 'Is_Anomaly']], use_container_width=True)
        export_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 이상탐지 결과 플래그가 포함된 전체 CSV 다운로드",
            data=export_csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

else:
    # 📌 최초 진입 시 가이드 UI 디자인 (Empty State 디자인)
    st.markdown(
        """
        <div style="border: 2px dashed #CBD5E1; padding: 60px 20px; border-radius: 12px; text-align: center; background-color: #F8FAFC; margin-top: 20px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">📥</div>
            <h3 style="color: #334155; font-weight:700; margin-bottom: 8px;">다변량 시계열 CSV 데이터 파일 대기 중</h3>
            <p style="color: #64748B; max-width: 580px; margin: 0 auto 25px auto; font-size: 0.95rem; line-height: 1.5;">
                분석을 원하시는 센서 및 로그 인프라의 임의 다변량 CSV 파일을 <b>왼쪽 사이드바 패널</b>에 드롭해 주세요. 파일이 탑재되면 대시보드가 자동으로 빌드됩니다.
            </p>
            <div style="font-size: 0.8rem; color: #94A3B8;">구조 조건: 시간 축 식별자 컬럼 필수 포함 + 연속성 수치 데이터 1개 이상 필수</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# 🏢 하단 푸터 영역 UX 추가
st.markdown('<div class="footer">© 2026 다변량 시계열 모니터링 시스템 구축 프로젝트 | 6/22 제출 최종 성과물</div>', unsafe_allow_html=True)
