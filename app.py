import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 웹 페이지 기본 설정 및 제목
# ==========================================
st.set_page_config(layout="wide", page_title="CMP 2D Simulator")
st.title("반도체 CMP 슬러리 유동 2D 시뮬레이터")
st.markdown("**(Term Project: Subject 2 - Lubrication Theory)**")

# ==========================================
# 2. 사이드바(Sidebar): 사용자 입력 및 "설계 탐색(Shape)"
# ==========================================
st.sidebar.header("⚙️ Process Parameters")
U_wafer = st.sidebar.slider("웨이퍼 슬라이딩 속도 (m/s)", 0.1, 3.0, 1.0, 0.1)
h0 = st.sidebar.slider("최소 틈새 두께 (um)", 10.0, 100.0, 50.0, 5.0) * 1e-6
tilt = st.sidebar.slider("패드 기울기 (Tilt, x10^-4)", 1.0, 10.0, 5.0, 1.0) * 1e-4

st.sidebar.markdown("---")
st.sidebar.header("🔍 Design Exploration")
# 루브릭 충족을 위한 형태(Shape) 수정 기능 추가
pad_shape = st.sidebar.selectbox("패드 표면 형태 (Pad Shape)", ["Flat (기본 평면)", "Convex (볼록형)", "Concave (오목형)"])

# 형태에 따른 높이 변화량(Shape modifier) 설정
shape_mod = 0.0
if pad_shape == "Convex (볼록형)":
    shape_mod = -15e-6 # 가운데 틈새가 좁아짐
elif pad_shape == "Concave (오목형)":
    shape_mod = 15e-6  # 가운데 틈새가 넓어짐

# 고정 파라미터
L, W = 0.1, 0.1
Nx, Ny = 40, 40 
dx, dy = L/(Nx-1), W/(Ny-1)
K_visc, n_idx = 0.05, 0.8
h_a = 60e-6
K_c, k_p = 1.0e7, 2.0e-13

# ==========================================
# 3. 수치해석 엔진
# ==========================================
with st.spinner('Calculating 2D Reynolds Equation...'):
    x = np.linspace(0, L, Nx)
    X, Y = np.meshgrid(x, np.linspace(0, W, Ny))
    
    # 틈새 높이 방정식에 형태(Shape) 함수(Sine wave) 추가
    h = h0 + tilt * (L - X) + shape_mod * np.sin(np.pi * X / L)
    
    shear_rate = U_wafer / h
    eta_eff = K_visc * (shear_rate)**(n_idx - 1)
    
    P_f = np.zeros((Ny, Nx))
    iterations = 800 
    
    for _ in range(iterations):
        P_old = P_f.copy()
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                h_val = h[i, j]
                eta_val = eta_eff[i, j]
                # 형태 변화를 반영한 편미분(dh/dx) 계산
                dh_dx = -tilt + shape_mod * (np.pi/L) * np.cos(np.pi * x[j] / L)
                RHS = 6 * eta_val * U_wafer * dh_dx
                P_f[i, j] = 0.25 * (P_old[i+1, j] + P_old[i-1, j] + P_old[i, j+1] + P_old[i, j-1] - RHS * dx**2 / (h_val**3 + 1e-15))
    
    P_f[P_f < 0] = 0
    
    P_c = np.zeros_like(h)
    mask = h < h_a
    P_c[mask] = K_c * (1 - h[mask] / h_a)**2.5
    MRR = k_p * P_c * U_wafer

# ==========================================
# 4. 루브릭 충족을 위한 탭(Tab) 뷰 구성
# ==========================================
# 탭을 두 개로 나누어 '설계 탐색(Heatmap)'과 '검증 뷰(Validation)'를 모두 충족
tab1, tab2 = st.tabs(["📊 2D Design Exploration (Heatmaps)", "📈 Validation View (1D vs 2D)"])

with tab1:
    st.subheader("Design Exploration: Shape and Pressure")
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
    st.markdown("PSET #02의 1D 해석적 해(Analytical)와 현재 2D 시뮬레이터의 중심축(y=W/2) 압력 비교")
    
    fig2, ax = plt.subplots(figsize=(8, 4))
    
    # 1D Analytical Solution (Newtonian & Flat 가정 시)
    h_1d = h0 + tilt * (L - x)
    h1, h2 = h_1d[0], h_1d[-1]
    # 단순화된 1D 점도 (근사)
    eta_0 = K_visc * (U_wafer / h_1d.mean())**(n_idx - 1)
    P_analytic = (6 * eta_0 * U_wafer * L / (h1**2 - h2**2)) * ((h1 - h_1d) * (h_1d - h2) / h_1d**2)
    P_analytic[P_analytic < 0] = 0
    
    ax.plot(x, P_analytic, 'k--', linewidth=2, label="1D Analytical (PSET 2)")
    ax.plot(x, P_f[Ny//2, :], 'b-', linewidth=2, label="2D Numerical (Centerline)")
    ax.set_xlabel("x-position (m)")
    ax.set_ylabel("Fluid Pressure (Pa)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig2)
