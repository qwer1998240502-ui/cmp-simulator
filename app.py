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
# 2. 사이드바(Sidebar): 사용자 입력 슬라이더
# ==========================================
st.sidebar.header("⚙️ Process Parameters")
U_wafer = st.sidebar.slider("웨이퍼 슬라이딩 속도 (m/s)", 0.1, 3.0, 1.0, 0.1)
h0 = st.sidebar.slider("최소 틈새 두께 (um)", 10.0, 100.0, 50.0, 5.0) * 1e-6
tilt = st.sidebar.slider("패드 기울기 (Tilt, x10^-4)", 1.0, 10.0, 5.0, 1.0) * 1e-4

# 고정 파라미터 (숨김 처리 또는 고정값)
L, W = 0.1, 0.1
Nx, Ny = 40, 40  # 웹에서 빠르게 돌아가도록 해상도를 살짝 낮춤
dx, dy = L/(Nx-1), W/(Ny-1)
K_visc, n_idx = 0.05, 0.8
h_a = 60e-6
K_c, k_p = 1.0e7, 2.0e-13

# ==========================================
# 3. 핵심 수치해석 엔진 (Step 2의 코드와 동일)
# ==========================================
# 사용자 경험을 위해 계산 중임을 알리는 스피너(Spinner) 표시
with st.spinner('Calculating 2D Reynolds Equation...'):
    X, Y = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, W, Ny))
    h = h0 + tilt * (L - X)
    
    shear_rate = U_wafer / h
    eta_eff = K_visc * (shear_rate)**(n_idx - 1)
    
    P_f = np.zeros((Ny, Nx))
    iterations = 800  # 빠른 실시간 반영을 위해 반복 횟수 조정
    
    for _ in range(iterations):
        P_old = P_f.copy()
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                h_val = h[i, j]
                eta_val = eta_eff[i, j]
                dh_dx = -tilt
                RHS = 6 * eta_val * U_wafer * dh_dx
                P_f[i, j] = 0.25 * (P_old[i+1, j] + P_old[i-1, j] + P_old[i, j+1] + P_old[i, j-1] - RHS * dx**2 / (h_val**3 + 1e-15))
    
    P_f[P_f < 0] = 0
    
    P_c = np.zeros_like(h)
    mask = h < h_a
    P_c[mask] = K_c * (1 - h[mask] / h_a)**2.5
    MRR = k_p * P_c * U_wafer

# ==========================================
# 4. 결과 시각화 (웹 페이지 중앙에 띄우기)
# ==========================================
st.subheader("📊 Simulation Results (Heatmaps)")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

c1 = axs[0].contourf(X, Y, P_f, 30, cmap='Blues')
axs[0].set_title('Fluid Pressure ($P_f$)')
fig.colorbar(c1, ax=axs[0])

c2 = axs[1].contourf(X, Y, P_c, 30, cmap='Oranges')
axs[1].set_title('Contact Pressure ($P_c$)')
fig.colorbar(c2, ax=axs[1])

c3 = axs[2].contourf(X, Y, MRR, 30, cmap='Greens')
axs[2].set_title('Material Removal Rate (MRR)')
fig.colorbar(c3, ax=axs[2])

# Streamlit 전용 함수를 사용해 웹페이지에 matplotlib 그래프를 출력
st.pyplot(fig)

st.success("계산이 완료되었습니다! 왼쪽 사이드바의 슬라이더를 움직여 결과를 실시간으로 확인해 보세요.")
