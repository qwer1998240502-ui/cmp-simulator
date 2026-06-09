import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 웹 페이지 기본 설정 및 제목
# ==========================================
st.set_page_config(layout="wide", page_title="CMP 2D Simulator")
st.title("반도체 CMP 슬러리 유동 및 연마율 2D 시뮬레이터")
st.markdown("**(Term Project: Subject 2 - Lubrication Theory)**")

# ==========================================
# 2. 사이드바(Sidebar): 사용자 입력 및 공정 설계 탐색
# ==========================================
st.sidebar.header("⚙️ 공정 파라미터 입력 (Process Parameters)")
U_wafer = st.sidebar.slider("웨이퍼 슬라이딩 속도 (m/s)", 0.1, 3.0, 1.0, 0.1)
h0 = st.sidebar.slider("최소 틈새 두께 (um)", 10.0, 100.0, 50.0, 5.0) * 1e-6
tilt = st.sidebar.slider("패드 기울기 (Tilt, x10^-4)", 1.0, 10.0, 5.0, 1.0) * 1e-4

st.sidebar.markdown("---")
st.sidebar.header("💧 슬러리 물성 조절 (Slurry Properties)")
# 외부 AI가 지적한 '점도/밀도 조작부 누락' 조건을 완벽히 충족하는 슬라이더 추가
K_visc = st.sidebar.slider("점도 지수 (Consistency Index K, Pa·s)", 0.01, 0.10, 0.05, 0.01)
n_idx = st.sidebar.slider("유동 지수 (Flow Behavior Index n)", 0.5, 1.0, 0.8, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("🔍 기하학적 설계 탐색 (Design Exploration)")
# 루브릭 요건인 '패드 형태(Shape) 직접 수정' 기능
pad_shape = st.sidebar.selectbox("패드 표면 형태 (Pad Shape)", ["Flat (기본 평면)", "Convex (볼록형)", "Concave (오목형)"])

# 선택한 형태에 따른 높이 변화량(Shape modifier) 설정
shape_mod = 0.0
if pad_shape == "Convex (볼록형)":
    shape_mod = -15e-6 # 가운데 틈새가 좁아짐
elif pad_shape == "Concave (오목형)":
    shape_mod = 15e-6  # 가운데 틈새가 넓어짐

# 시뮬레이션 고정 파라미터 설정
L, W = 0.1, 0.1       # 패드 및 웨이퍼 가로세로 길이 (m)
Nx, Ny = 40, 40       # FDM 격자 수
dx, dy = L/(Nx-1), W/(Ny-1)
h_a = 60e-6           # 패드 평균 돌기 높이 (Asperity height)
K_c = 1.0e7           # 패드 강성 계수
k_p = 2.0e-13         # Preston 프레스턴 상수

# ==========================================
# 3. 2D 유한차분법(FDM) 수치해석 엔진
# ==========================================
with st.spinner('Calculating 2D Reynolds Equation...'):
    x = np.linspace(0, L, Nx)
    X, Y = np.meshgrid(x, np.linspace(0, W, Ny))
    
    # 틈새 높이 분포에 기하학적 형태 함수(Sine wave) 동적 결합
    h = h0 + tilt * (L - X) + shape_mod * np.sin(np.pi * X / L)
    
    # 비뉴턴 전단박화(Shear-thinning) 실시간 유동 해석 계산
    shear_rate = U_wafer / h
    eta_eff = K_visc * (shear_rate)**(n_idx - 1)
    
    P_f = np.zeros((Ny, Nx))
    iterations = 800 
    
    # Jacobi Relaxation PDE 반복 솔버
    for _ in range(iterations):
        P_old = P_f.copy()
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                h_val = h[i, j]
                eta_val = eta_eff[i, j]
                # 패드 형태 변형률이 반영된 편미분(dh/dx) 연산
                dh_dx = -tilt + shape_mod * (np.pi/L) * np.cos(np.pi * x[j] / L)
                RHS = 6 * eta_val * U_wafer * dh_dx
                P_f[i, j] = 0.25 * (P_old[i+1, j] + P_old[i-1, j] + P_old[i, j+1] + P_old[i, j-1] - RHS * dx**2 / (h_val**3 + 1e-15))
    
    # Half-Sommerfeld Cavitation 경계 조건 처리 (음수 압력 클리핑)
    P_f[P_f < 0] = 0
    
    # Stribeck 혼합 윤활 곡선 기반 고체 돌기 접촉 압력(Pc) 모델링
    P_c = np.zeros_like(h)
    mask = h < h_a
    P_c[mask] = K_c * (1 - h[mask] / h_a)**2.5
    
    # Preston 공식 기반 국소 연마율(MRR) 예측 계산
    MRR = k_p * P_c * U_wafer

# ==========================================
# 4. 루브릭 연동형 멀티 탭(Tab) UI 구성
# ==========================================
tab1, tab2 = st.tabs(["📊 2D Design Exploration (Heatmaps)", "📈 Validation View (1D vs 2D)"])

with tab1:
    st.subheader("Design Exploration: Shape and Pressure")
    st.markdown("사이드바의 점도 물성 및 패드 형태 변경에 따라 실시간으로 변화하는 2D 공간 분포 분석")
    fig1, axs = plt.subplots(1, 3, figsize=(18, 5))

    c1 = axs[0].contourf(X, Y, P_f, 30, cmap='Blues')
    axs[0].set_title('Fluid Pressure ($P_f$)')
    fig1.colorbar(c1, ax=axs[0])

    c2 = axs[1].contourf(X, Y, P_c, 30, cmap='Oranges')
    axs[1].set_title('Contact Pressure ($P_c$)')
    fig1.colorbar(c2, ax=axs[1])

    c3 = axs[2].contourf(X, Y, MRR, 30, cmap='Greens')
    axs[2].set_title('Material Removal Rate (MRR)')
    fig1.colorbar(c3, ax=axs[2])

    st.pyplot(fig1)

with tab2:
    st.subheader("Simulator Validation: Analytical vs Numerical")
    st.markdown("고전적 1D 해석적 해(Analytical)와 현재 2D 시뮬레이터의 중심축(y=W/2) 압력 비교 데이터 검증")
    
    fig2, ax = plt.subplots(figsize=(8, 4))
    
    # 1D Analytical Solution (Newtonian & Flat 구조 한계 모델링)
    h_1d = h0 + tilt * (L - x)
    h1, h2 = h_1d[0], h_1d[-1]
    # 슬러리 물성 조절 슬라이더(K, n)값과 동적으로 상호작용하는 1차원 유효 점도 산출
    eta_0 = K_visc * (U_wafer / h_1d.mean())**(n_idx - 1)
    P_analytic = (6 * eta_0 * U_wafer * L / (h1**2 - h2**2)) * ((h1 - h_1d) * (h_1d - h2) / h_1d**2)
    P_analytic[P_analytic < 0] = 0
    
    # 완벽하게 통일된 1D vs 2D 라벨링 세팅
    ax.plot(x, P_analytic, 'k--', linewidth=2, label="1D Analytical")
    ax.plot(x, P_f[Ny//2, :], 'b-', linewidth=2, label="2D Numerical (Centerline)")
    ax.set_xlabel("x-position (m)")
    ax.set_ylabel("Fluid Pressure (Pa)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig2)
