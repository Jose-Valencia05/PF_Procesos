"""
Modulo de Cadenas de Markov — Resultados W/L
==============================================
Modela los resultados de partidos (W/L) como una cadena de Markov
de 2 estados. Calcula:

- Matriz de transicion empirica P(W|W), P(L|W), P(W|L), P(L|L)
- Distribucion estacionaria (proporcion de victorias a largo plazo)
- Prediccion del siguiente partido
- Grafica del diagrama de transicion

Autor: Proyecto Procesos Estocasticos - NBA
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 120,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def construir_matriz_transicion(resultados):
    """
    Construye la matriz de transicion empirica de 2 estados: W y L.

    Parameters
    ----------
    resultados : array-like
        Secuencia de resultados ('W' o 'L') en orden cronologico.

    Returns
    -------
    tuple
        (matriz P 2x2, labels ['W','L'], conteos dict)
        P[i][j] = P(siguiente=j | actual=i)
    """
    res = np.array(resultados)
    n = len(res)

    conteos = {"W->W": 0, "W->L": 0, "L->W": 0, "L->L": 0}

    for t in range(n - 1):
        actual = res[t]
        siguiente = res[t + 1]
        conteos[f"{actual}->{siguiente}"] += 1

    total_W = conteos["W->W"] + conteos["W->L"]
    total_L = conteos["L->W"] + conteos["L->L"]

    P = np.zeros((2, 2))
    labels = ["W", "L"]

    if total_W > 0:
        P[0, 0] = conteos["W->W"] / total_W
        P[0, 1] = conteos["W->L"] / total_W
    if total_L > 0:
        P[1, 0] = conteos["L->W"] / total_L
        P[1, 1] = conteos["L->L"] / total_L

    return P, labels, conteos


def distribucion_estacionaria(P):
    """
    Calcula la distribucion estacionaria pi resolviendo pi * P = pi.

    Para una cadena de 2 estados con matriz P = [[a, 1-a], [b, 1-b]]:
    pi_W = b / (1 - a + b)
    pi_L = (1 - a) / (1 - a + b)

    Parameters
    ----------
    P : numpy.ndarray
        Matriz de transicion 2x2.

    Returns
    -------
    numpy.ndarray
        Vector pi = [pi_W, pi_L].
    """
    a = P[0, 0]
    b = P[1, 0]

    denom = (1 - a) + b
    if abs(denom) < 1e-12:
        denom = 1.0

    pi_W = b / denom
    pi_L = (1 - a) / denom
    return np.array([pi_W, pi_L])


def predecir_siguiente(resultados, P):
    """
    Predice el resultado del siguiente partido usando la matriz
    de transicion y el ultimo resultado observado.

    Parameters
    ----------
    resultados : array-like
        Secuencia de resultados W/L.
    P : numpy.ndarray
        Matriz de transicion 2x2.

    Returns
    -------
    dict
        Prediccion con probabilidades.
    """
    ultimo = resultados[-1]
    idx = 0 if ultimo == "W" else 1
    prob_W = P[idx, 0]
    prob_L = P[idx, 1]
    prediccion = "W" if prob_W >= prob_L else "L"

    return {
        "ultimo_resultado": ultimo,
        "probabilidad_W": prob_W,
        "probabilidad_L": prob_L,
        "prediccion": prediccion,
    }


def graficar_transiciones(P, equipo="Equipo", plots_dir=None):
    """
    Grafica el diagrama de transicion de la cadena de Markov.

    Parameters
    ----------
    P : numpy.ndarray
        Matriz de transicion 2x2.
    equipo : str
        Nombre del equipo.
    plots_dir : str, optional
        Directorio donde guardar la grafica.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    r = 3
    t_W = np.pi / 2
    t_L = 3 * np.pi / 2
    pos = {"W": (r * np.cos(t_W), r * np.sin(t_W)),
           "L": (r * np.cos(t_L), r * np.sin(t_L))}

    for label, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.65, color="#2196F3", ec="white",
                             linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=20,
                fontweight="bold", color="white", zorder=4)

    for origen, x0, y0, dx, dy, off, color in [
        ("W", pos["W"][0], pos["W"][1],
         pos["W"][0] - pos["L"][0], pos["W"][1] - pos["L"][1], 13, "#4CAF50"),
        ("L", pos["L"][0], pos["L"][1],
         pos["L"][0] - pos["W"][0], pos["L"][1] - pos["W"][1], -13, "#FF9800"),
    ]:
        angle = np.arctan2(dy, dx)
        x0_adj = x0 - 0.65 * np.cos(angle)
        y0_adj = y0 - 0.65 * np.sin(angle)
        x1_adj = x0 + dx + 0.65 * np.cos(angle)
        y1_adj = y0 + dy + 0.65 * np.sin(angle)

        ax.annotate("", xy=(x1_adj, y1_adj), xytext=(x0_adj, y0_adj),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2.5,
                                     connectionstyle="arc3,rad=.2"))

        mid_x = (x0 + dx / 2) + off * np.cos(angle + np.pi / 2)
        mid_y = (y0 + dy / 2) + off * np.sin(angle + np.pi / 2)
        prob = P[0, 1] if origen == "W" else P[1, 0]
        ax.text(mid_x, mid_y, f"{prob:.3f}", fontsize=13,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.85))

    for i, (label, (x, y)) in enumerate(pos.items()):
        prob = P[i, i]
        angle_self = np.pi / 4 if label == "W" else -np.pi / 4
        loop_r = 1.0
        cx = x + (0.65 + loop_r) * np.cos(angle_self)
        cy = y + (0.65 + loop_r) * np.sin(angle_self)

        theta = np.linspace(angle_self - 1.0, angle_self + 1.0, 60)
        lx = x + (0.65) * np.cos(theta)
        ly = y + (0.65) * np.sin(theta)

        anchor_angle = angle_self + 0.6 if label == "W" else angle_self - 0.6
        ax.plot(lx, ly, color="#D32F2F", linewidth=2.2, alpha=0.7)
        ax.annotate("", xy=(lx[-1], ly[-1]), xytext=(lx[-2], ly[-2]),
                     arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2.2))

        tx = x + (0.65 + loop_r + 0.35) * np.cos(angle_self)
        ty = y + (0.65 + loop_r + 0.35) * np.sin(angle_self)
        ax.text(tx, ty, f"{prob:.3f}", fontsize=12,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.85))

    ax.text(0.02, 0.97,
            "Diagrama de transicion de la cadena de Markov.\n"
            "Muestra las probabilidades de pasar de un estado\n"
            "(W=Victoria, L=Derrota) al siguiente.",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlim(-7, 7)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_title(f"Cadena de Markov: Resultados W/L — {equipo}")
    ax.axis("off")

    plt.tight_layout()

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    filepath = os.path.join(plots_dir, "markov_transiciones.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafica guardada: {filepath}")
    return filepath


def markov(df, equipo="Warriors", plots_dir=None):
    """
    Funcion principal del modulo de Markov.
    Ejecuta el analisis completo de cadena de Markov W/L.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame limpio con columna 'resultado' (W/L).
    equipo : str
        Nombre del equipo.
    plots_dir : str, optional
        Directorio donde guardar las graficas.

    Returns
    -------
    dict
        Resultados del analisis de Markov.
    """
    print(f"\n{'='*60}")
    print(f"  CADENA DE MARKOV (W/L) — {equipo}")
    print(f"{'='*60}")

    col_res = "resultado" if "resultado" in df.columns else "WL"
    resultados = df[col_res].values

    P, labels, conteos = construir_matriz_transicion(resultados)
    pi = distribucion_estacionaria(P)
    prediccion = predecir_siguiente(resultados, P)

    print(f"\n  Matriz de transicion:")
    print(f"  P(W|W) = {P[0,0]:.3f}    P(L|W) = {P[0,1]:.3f}")
    print(f"  P(W|L) = {P[1,0]:.3f}    P(L|L) = {P[1,1]:.3f}")
    print(f"\n  Conteos de transiciones:")
    print(f"  W -> W: {conteos['W->W']}    W -> L: {conteos['W->L']}")
    print(f"  L -> W: {conteos['L->W']}    L -> L: {conteos['L->L']}")
    print(f"\n  Distribucion estacionaria pi:")
    print(f"  pi_W (victorias a largo plazo) = {pi[0]:.3f}")
    print(f"  pi_L (derrotas a largo plazo)   = {pi[1]:.3f}")
    print(f"  Proporcion observada de W: {np.mean(resultados == 'W'):.3f}")
    print(f"\n  Prediccion siguiente partido:")
    print(f"  Ultimo resultado: {prediccion['ultimo_resultado']}")
    print(f"  P(siguiente = W) = {prediccion['probabilidad_W']:.3f}")
    print(f"  P(siguiente = L) = {prediccion['probabilidad_L']:.3f}")
    print(f"  Prediccion: {prediccion['prediccion']}")

    if plots_dir:
        graficar_transiciones(P, equipo=equipo, plots_dir=plots_dir)

    return {
        "matriz_transicion": P.tolist(),
        "conteos": conteos,
        "pi_W": pi[0],
        "pi_L": pi[1],
        "proporcion_W_observada": float(np.mean(resultados == "W")),
        "ultimo_resultado": prediccion["ultimo_resultado"],
        "probabilidad_W_siguiente": prediccion["probabilidad_W"],
        "prediccion": prediccion["prediccion"],
    }
