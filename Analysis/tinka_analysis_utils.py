# =====================
# Funciones
# =====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random


def choose_random_numbers(quantity=6, total_numbers=50):
    """
    Choose a given quantity of unique random numbers from 1 to total_numbers.
    
    Parameters:
        quantity (int): How many numbers to choose (minimum 6).
        total_numbers (int): Range of numbers to draw from (1 to total_numbers).
    
    Returns:
        list: Sorted list of chosen numbers.
    """
    if quantity < 6:
        raise ValueError("You must choose at least 6 numbers.")
    return sorted(random.sample(range(1, total_numbers + 1), quantity))


def simulate_tinka(include_boliyapa=True, total_numbers=50):
    """
    Simulates a Tinka draw with optional Boliyapa.
    
    Parameters:
        include_boliyapa (bool): If True, adds a Boliyapa as the last number in the list.
        total_numbers (int): Range of numbers to draw from (1 to total_numbers).
    
    Returns:
        list: Sorted list of 6 numbers, plus Boliyapa at the end if included.
    """
    first_six = sorted(random.sample(range(1, total_numbers + 1), 6))
    if include_boliyapa:
        boliyapa = random.choice([n for n in range(1, total_numbers + 1) if n not in first_six])
        return first_six + [boliyapa]
    return first_six


def plot_numbers(my_numbers, winning_numbers):
    """
    Plots two lists of numbers vertically aligned, marking matches in red and 
    connecting them with precise arrows.

    Parameters:
        my_numbers (list): Player's chosen numbers (6 or more).
        winning_numbers (list): Winning numbers (6 or 7 if Boliyapa included).
    """
    matches = set(my_numbers) & set(winning_numbers)
    max_len = max(len(my_numbers), len(winning_numbers))
    
    plt.figure(figsize=(10, 4))
    
    # Graficar mis números (fila superior)
    for idx, num in enumerate(my_numbers):
        color = 'red' if num in matches else 'blue'
        plt.scatter(idx, 1, color=color, s=300, edgecolors='black', zorder=3)
        plt.text(idx, 1.15, str(num), ha='center', fontsize=12, fontweight='bold', zorder=4)
    
    # Graficar números ganadores (fila inferior)
    for idx, num in enumerate(winning_numbers):
        color = 'red' if num in matches else 'blue'
        plt.scatter(idx, 0, color=color, s=300, edgecolors='black', zorder=3)
        plt.text(idx, -0.25, str(num), ha='center', fontsize=12, fontweight='bold', zorder=4)
    
    # Dibujar flechas exactas para coincidencias
    for my_idx, my_num in enumerate(my_numbers):
        if my_num in matches:
            for win_idx, win_num in enumerate(winning_numbers):
                if my_num == win_num:
                    plt.annotate(
                        "", 
                        xy=(win_idx, 0.05),      # Desde el centro del círculo ganador
                        xytext=(my_idx, 0.95),   # Desde el centro del círculo del jugador
                        arrowprops=dict(arrowstyle="->", color="red", lw=2),
                        zorder=2
                    )

    plt.yticks([0, 1], ["Winning Numbers", "My Numbers"])
    plt.ylim(-0.5, 1.5)
    plt.title(f"Matches: {len(matches)}", fontsize=14, pad=20)
    plt.xticks(range(max_len), [])
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def monte_carlo_tinka(my_numbers, wins_needed=6, simulations=1000, include_boliyapa=False):
    """
    Runs a Monte Carlo simulation to estimate the probability of winning.
    
    Parameters:
        my_numbers (list): Player's chosen numbers (6 or more).
        wins_needed (int): Number of matches needed to be considered a win.
        simulations (int): Number of simulated draws.
        include_boliyapa (bool): If True, include Boliyapa in draws.
    
    Returns:
        None: Shows a line plot of the frequency of wins over simulations.
    """
    win_counts = []
    
    for _ in range(simulations):
        winning_numbers = simulate_tinka(include_boliyapa=include_boliyapa)
        matches = len(set(my_numbers) & set(winning_numbers))
        win_counts.append(matches >= wins_needed)
    
    cumulative_wins = [sum(win_counts[:i+1]) for i in range(simulations)]
    win_percentages = [wins/(i+1) for i, wins in enumerate(cumulative_wins)]
    
    final_probability = win_percentages[-1]  # Probabilidad final después de todas las simulaciones
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, simulations+1), win_percentages, label=f"Win probability (≥ {wins_needed} matches)")
    plt.xlabel("Simulations")
    plt.ylabel("Probability")
    plt.title("Monte Carlo Simulation - La Tinka")
    plt.legend()
    plt.grid(True)
    
    # Mostrar la probabilidad
    plt.text(simulations * 0.7, min(win_percentages) + (max(win_percentages) - min(win_percentages)) * 0.05,
             f"Probability: {final_probability:.10f}",
             fontsize=12, color="red",
             ha="right", bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.show()


def analyze_number_frequency(df):
    """
    Analiza la frecuencia de aparición de cada número en el histórico de la Tinka.
    
    Parameters:
        df (DataFrame): DataFrame con columnas 'fecha', 'bola', 'valor'
    
    Returns:
        DataFrame: Tabla con número y su frecuencia.
    """
    freq = df[df["bola"] != "Boliyapa"]["valor"].value_counts().sort_index()
    
    plt.figure(figsize=(10, 4))
    plt.bar(freq.index, freq.values)
    plt.xlabel("Número")
    plt.ylabel("Frecuencia")
    plt.title("Frecuencia de aparición de números (sin Boliyapa)")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()
    
    return pd.DataFrame({"Número": freq.index, "Frecuencia": freq.values})

def hot_and_cold_numbers(df, recent_draws=10):
    """
    Encuentra números calientes y fríos en los últimos sorteos.
    
    Parameters:
        df (DataFrame): Histórico de la Tinka en formato largo.
        recent_draws (int): Número de sorteos recientes a considerar.
    
    Returns:
        tuple: (números calientes, números fríos)
    """
    recent_dates = df["fecha"].drop_duplicates().sort_values(ascending=False).head(recent_draws)
    recent_data = df[df["fecha"].isin(recent_dates)]
    
    freq_recent = recent_data[recent_data["bola"] != "Boliyapa"]["valor"].value_counts()
    hot_numbers = freq_recent.head(5).index.tolist()
    cold_numbers = freq_recent.tail(5).index.tolist()
    
    print(f"Números calientes (últimos {recent_draws} sorteos): {hot_numbers}")
    print(f"Números fríos (últimos {recent_draws} sorteos): {cold_numbers}")
    
    return hot_numbers, cold_numbers

def analyze_even_odd(df):
    """
    Analiza cuántos pares e impares suelen salir en cada sorteo.
    
    Parameters:
        df (DataFrame): Histórico de la Tinka.
    
    Returns:
        Series: Conteo de ocurrencias por cantidad de pares.
    """
    non_boliyapa = df[df["bola"] != "Boliyapa"]
    even_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n % 2 == 0 for n in x))
    
    dist = even_counts.value_counts().sort_index()
    plt.bar(dist.index, dist.values)
    plt.xlabel("Cantidad de números pares")
    plt.ylabel("Frecuencia de sorteos")
    plt.title("Distribución de pares en sorteos")
    plt.show()
    
    return dist

def analyze_low_high(df, split_point=25):
    """
    Analiza cuántos números bajos y altos suelen salir en cada sorteo.
    
    Parameters:
        df (DataFrame): Histórico de la Tinka.
        split_point (int): Número que divide el rango bajo/alto.
    
    Returns:
        Series: Conteo de ocurrencias por cantidad de bajos.
    """
    non_boliyapa = df[df["bola"] != "Boliyapa"]
    low_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n <= split_point for n in x))
    
    dist = low_counts.value_counts().sort_index()
    plt.bar(dist.index, dist.values)
    plt.xlabel(f"Cantidad de números ≤ {split_point}")
    plt.ylabel("Frecuencia de sorteos")
    plt.title("Distribución de números bajos en sorteos")
    plt.show()
    
    return dist

def analyze_repeated_sequences(df_long):
    """
    Analiza cuántas veces se repiten combinaciones exactas de 6 números en el histórico.
    """
    combos = (df_long[df_long["bola"] != "Boliyapa"]
              .groupby("fecha")["valor"]
              .apply(lambda x: tuple(sorted(x))))
    
    combo_counts = combos.value_counts()
    repeated = combo_counts[combo_counts > 1]
    
    if repeated.empty:
        print("No hay combinaciones repetidas en el histórico.")
    else:
        print("Combinaciones repetidas:\n", repeated)

def analyze_consecutive_numbers(df_long):
    """
    Analiza cuántos pares de números consecutivos aparecen en cada sorteo.
    """
    counts = []
    
    for _, group in df_long[df_long["bola"] != "Boliyapa"].groupby("fecha"):
        nums = sorted(group["valor"])
        consecutive_pairs = sum((b - a) == 1 for a, b in zip(nums, nums[1:]))
        counts.append(consecutive_pairs)
    
    plt.figure(figsize=(6,4))
    sns.countplot(x=counts, palette="viridis")
    plt.title("Cantidad de pares consecutivos por sorteo")
    plt.xlabel("Pares consecutivos")
    plt.ylabel("Frecuencia")
    plt.show()

def analyze_transitions(df_long, top_n=10):
    """
    Muestra una matriz de transición simple: frecuencia con la que un número aparece
    después de otro en el siguiente sorteo.
    """
    df_main = df_long[df_long["bola"] != "Boliyapa"].copy()
    df_main = df_main.sort_values("fecha")
    
    # Agrupar por fecha
    draws = df_main.groupby("fecha")["valor"].apply(set).tolist()
    
    transitions = Counter()
    for prev, curr in zip(draws, draws[1:]):
        for p in prev:
            for c in curr:
                transitions[(p, c)] += 1
    
    # Pasar a DataFrame
    df_trans = pd.DataFrame([
        {"prev": p, "curr": c, "count": cnt}
        for (p, c), cnt in transitions.items()
    ])
    
    top_pairs = df_trans.sort_values("count", ascending=False).head(top_n)
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=top_pairs, x="count", y=top_pairs.apply(lambda r: f"{r.prev}→{r.curr}", axis=1), palette="magma")
    plt.title(f"Top {top_n} transiciones más frecuentes entre sorteos")
    plt.xlabel("Frecuencia")
    plt.ylabel("Transición")
    plt.show()


def analyze_rare_numbers(df_long, top_n=10):
    """
    Muestra los números menos frecuentes en el histórico.
    """
    nums = df_long[df_long["bola"] != "Boliyapa"]["valor"]
    freq = nums.value_counts().sort_values().head(top_n)
    
    plt.figure(figsize=(8,4))
    sns.barplot(x=freq.index, y=freq.values, palette="coolwarm")
    plt.title(f"Top {top_n} números menos frecuentes")
    plt.xlabel("Número")
    plt.ylabel("Frecuencia")
    plt.show()


def analyze_repeats(df_tinka):
    """
    Analiza cuántos números ganadores se repiten de un sorteo al siguiente.
    
    Parámetros:
        df_tinka (pd.DataFrame): DataFrame con columnas ['fecha', 'bola', 'valor'].
                                 Debe contener los datos históricos de la Tinka.
                                 
    Retorna:
        pd.DataFrame: Conteo y porcentaje de coincidencias.
    """
    # Filtramos solo las 6 bolas principales (sin Boliyapa)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Agrupamos por fecha y creamos listas de números sorteados
    draws = df_main.groupby('fecha')['valor'].apply(list).reset_index()
    draws = draws.sort_values('fecha').reset_index(drop=True)
    
    repeat_counts = []

    # Comparar cada sorteo con el anterior
    for i in range(1, len(draws)):
        prev_set = set(draws.loc[i-1, 'valor'])
        curr_set = set(draws.loc[i, 'valor'])
        matches = len(prev_set & curr_set)
        repeat_counts.append(matches)
    
    # Contar ocurrencias
    count_distribution = Counter(repeat_counts)
    
    # Convertir a DataFrame para mostrar
    df_counts = pd.DataFrame({
        'Coincidencias': list(count_distribution.keys()),
        'Frecuencia': list(count_distribution.values())
    }).sort_values('Coincidencias').reset_index(drop=True)
    
    # Calcular porcentaje
    total_compared = len(repeat_counts)
    df_counts['Porcentaje'] = (df_counts['Frecuencia'] / total_compared * 100).round(2)
    
    # Graficar histograma
    plt.figure(figsize=(8, 5))
    plt.bar(df_counts['Coincidencias'], df_counts['Frecuencia'], color='skyblue', edgecolor='black')
    plt.xlabel("Cantidad de coincidencias con el sorteo anterior")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de coincidencias entre sorteos consecutivos")
    
    # Mostrar porcentajes sobre las barras
    for i, (x, y, pct) in enumerate(zip(df_counts['Coincidencias'], df_counts['Frecuencia'], df_counts['Porcentaje'])):
        plt.text(x, y + 0.5, f"{pct}%", ha='center', fontsize=9)
    
    plt.xticks(range(0, 7))  # Solo valores 0 a 6
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    return df_counts



def analyze_ball_position_ranges(df_tinka, range_step=10):
    """
    Analiza la distribución de cada bola por su posición (ordenada de menor a mayor)
    y muestra porcentajes de aparición por rangos.
    
    Parámetros:
        df_tinka (pd.DataFrame): DataFrame con columnas ['fecha', 'bola', 'valor'].
                                 Debe contener los datos históricos de la Tinka.
        range_step (int): Tamaño del rango para agrupar valores (por defecto 10).
    
    Retorna:
        dict: Diccionario con estadísticas de porcentajes por rango para cada posición.
    """
    # Filtrar solo bolas principales (B1 a B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Agrupar por fecha y ordenar cada sorteo
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    # Crear estructura para almacenar rangos
    ball_positions = {f"Bola {i+1}": [] for i in range(6)}
    
    # Llenar con valores por posición
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Bola {i+1}"].append(val)
    
    stats_by_position = {}
    
    plt.figure(figsize=(14, 8))
    
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        
        # Calcular rangos
        max_val = values_arr.max()
        bins = list(range(1, max_val + range_step, range_step))
        hist, bin_edges = np.histogram(values_arr, bins=bins)
        
        # Calcular porcentajes
        percentages = (hist / hist.sum() * 100).round(2)
        
        # Guardar en diccionario
        stats_df = pd.DataFrame({
            'Rango': [f"{bin_edges[j]}-{bin_edges[j+1]-1}" for j in range(len(hist))],
            'Frecuencia': hist,
            'Porcentaje': percentages
        })
        stats_by_position[ball_name] = stats_df
        
        # Graficar
        plt.subplot(2, 3, i)
        plt.bar(stats_df['Rango'], stats_df['Porcentaje'], color='skyblue', edgecolor='black')
        plt.title(f"{ball_name} - Distribución por rangos")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Distribución de cada bola por posición y rango", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_optimal_ranges_per_position(df_tinka, bins=5):
    """
    Analiza la distribución de cada bola por posición (ordenada) y determina rangos óptimos.
    
    Parámetros:
        df_tinka (pd.DataFrame): DataFrame con columnas ['fecha', 'bola', 'valor'].
        bins (int): Número de rangos a crear por posición (cuantiles).
    
    Retorna:
        dict: Rangos óptimos por posición con porcentajes.
    """
    # Filtrar solo bolas principales (B1 a B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Agrupar por fecha y ordenar cada sorteo
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    ball_positions = {f"Bola {i+1}": [] for i in range(6)}
    
    # Llenar listas de cada posición
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Bola {i+1}"].append(val)
    
    stats_by_position = {}
    
    plt.figure(figsize=(14, 8))
    
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        
        # Calcular cuantiles para cortes óptimos
        quantile_edges = np.linspace(0, 1, bins+1)
        edges = np.unique(np.percentile(values_arr, quantile_edges * 100).astype(int))
        
        # Aseguramos que los bordes cubran todo
        edges[0] = 1
        edges[-1] = 50
        
        hist, bin_edges = np.histogram(values_arr, bins=edges)
        percentages = (hist / hist.sum() * 100).round(2)
        
        stats_df = pd.DataFrame({
            'Rango': [f"{bin_edges[j]}-{bin_edges[j+1]}" for j in range(len(hist))],
            'Frecuencia': hist,
            'Porcentaje': percentages
        })
        stats_by_position[ball_name] = stats_df
        
        # Graficar
        plt.subplot(2, 3, i)
        plt.bar(stats_df['Rango'], stats_df['Porcentaje'], color='lightgreen', edgecolor='black')
        plt.title(f"{ball_name} - Rangos óptimos")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Distribución y rangos óptimos por posición", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_histograms_per_position(df_tinka):
    """
    Muestra histogramas independientes para cada bola por posición.
    
    Parámetros:
        df_tinka (pd.DataFrame): DataFrame con columnas ['fecha', 'bola', 'valor'].
    
    Retorna:
        dict: Lista de valores por posición.
    """
    # Filtrar solo bolas principales (B1 a B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Agrupar por fecha y ordenar cada sorteo
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    ball_positions = {f"Bola {i+1}": [] for i in range(6)}
    
    # Llenar listas de cada posición
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Bola {i+1}"].append(val)
    
    # Graficar histogramas
    plt.figure(figsize=(14, 8))
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        plt.subplot(2, 3, i)
        plt.hist(values, bins=range(1, 52), color='skyblue', edgecolor='black', align='left')
        plt.title(f"{ball_name} - Histograma")
        plt.xlabel("Número")
        plt.ylabel("Frecuencia")
        plt.xticks(range(1, 51, 2))  # Números del 1 al 50 cada 2 para que no se amontonen
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Distribución histórica por posición (Bolas ordenadas)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


