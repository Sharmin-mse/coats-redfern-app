import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score, mean_absolute_error
from io import BytesIO

st.set_page_config(page_title="Coats-Redfern Kinetic Model Fitting", layout="centered")

st.title("🔥 Coats-Redfern Kinetic Model Fitting Tool")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
beta = st.number_input("Enter Heating Rate (β) in K/min", value=20.0)

model_dict = {
    "D2: 2D Diffusion": lambda a: (1 - a) * np.log(1 - a) + a,
    "D3: 3D Diffusion (Jander)": lambda a: (1 - (1 - a) ** (1/3)) ** 2,
    "D4: 3D Diffusion (Ginstling–Brounshtein)": lambda a: 1 - (2 * a / 3) - (1 - a)**(2 / 3),
    "A2: Avrami-Erofeev (n=2)": lambda a: (-np.log(1 - np.clip(a, 0, 0.99999))) ** 0.5,
    "A3: Avrami-Erofeev (n=3)": lambda a: (-np.log(1 - np.clip(a, 0, 0.99999))) ** (1 / 3),
    "R2: Contracting Cylinder": lambda a: 1 - (1 - a) ** (1 / 2),
    "R3: Contracting Sphere": lambda a: 1 - (1 - a) ** (1 / 3),
    "C1: First Order": lambda a: -np.log(1 - np.clip(a, 0, 0.99999)),
    "C2: Second Order": lambda a: (1 - a)**-1 - 1,
    "P2: Power Law (n=2)": lambda a: a ** 0.5,
    "P3: Power Law (n=3)": lambda a: a ** (1/3),
    "P4: Power Law (n=4)": lambda a: a ** (1/4)
}

model_name = st.selectbox("Choose a Reaction Model", list(model_dict.keys()))

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    T = df["Temperature/K"].values
    alpha = df["Degree of Reaction (α)"].values

    valid = (alpha > 0.1) & (alpha < 0.9)
    T = T[valid]
    alpha = alpha[valid]

    g_alpha = model_dict[model_name](alpha)
    y = np.log(g_alpha / T**2)

    def coats_redfern(T, E_a, A):
        T_mean = np.mean(T)
        term_inside_log = 1 - ((2 * 8.314 * T_mean) / E_a)
        log_argument = ((A * 8.314) / (E_a * beta)) * term_inside_log
        log_argument = np.clip(log_argument, 1e-20, None)
        ln_term = np.log(log_argument)
        return - (E_a / 8.314) * (1 / T) + ln_term

    try:
        params, _ = opt.curve_fit(coats_redfern, T, y, p0=[150000, 1e10])
        E_fit, A_fit = params
        E_fit_kJ = E_fit / 1000

        y_fit = coats_redfern(T, E_fit, A_fit)
        r2 = r2_score(y, y_fit)
        mae = mean_absolute_error(y, y_fit)
        mape = np.mean(np.abs((y - y_fit) / y)) * 100

        fig, ax = plt.subplots()
        ax.scatter(1/T, y, label='Data', color='red', alpha=0.6)
        ax.plot(1/T, y_fit, label='Fit', color='blue')
        ax.set_xlabel("1 / T (K⁻¹)")
        ax.set_ylabel("ln[g(α)/T²]")
        ax.set_title(f"Coats-Redfern Fit ({model_name})")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.markdown(f"**Activation Energy (Eₐ):** {E_fit_kJ:.2f} kJ/mol")
        st.markdown(f"**Pre-exponential Factor (A):** {A_fit:.2e} 1/min")
        st.markdown(f"**R²:** {r2:.4f}")
        st.markdown(f"**MAE:** {mae:.4f}")
        st.markdown(f"**MAPE:** {mape:.2f}%")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button(
            label="📥 Download Plot as PNG",
            data=buf.getvalue(),
            file_name="coats_redfern_plot.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error during fitting: {e}")
