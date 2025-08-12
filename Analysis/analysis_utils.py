# =====================
# Functions
# =====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import numpy as np


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
    
    # Plot my numbers (top row)
    for idx, num in enumerate(my_numbers):
        color = 'yellow' if num in matches else 'blue'
        plt.scatter(idx, 1, color=color, s=300, edgecolors='black', zorder=3)
        plt.text(idx, 1.15, str(num), ha='center', fontsize=12, fontweight='bold', zorder=4)
    
    # Plot winning numbers (bottom row)
    for idx, num in enumerate(winning_numbers):
        color = 'yellow' if num in matches else 'blue'
        plt.scatter(idx, 0, color=color, s=300, edgecolors='black', zorder=3)
        plt.text(idx, -0.25, str(num), ha='center', fontsize=12, fontweight='bold', zorder=4)
    
    # Draw exact arrows for matches
    for my_idx, my_num in enumerate(my_numbers):
        if my_num in matches:
            for win_idx, win_num in enumerate(winning_numbers):
                if my_num == win_num:
                    plt.annotate(
                        "", 
                        xy=(win_idx, 0.05),      # From the center of the winning circle
                        xytext=(my_idx, 0.95),   # From the center of the player's circle
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
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
    
    final_probability = win_percentages[-1]  # Final probability after all simulations
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, simulations+1), win_percentages, label=f"Win probability (≥ {wins_needed} matches)")
    plt.xlabel("Simulations")
    plt.ylabel("Probability")
    plt.title("Monte Carlo Simulation - La Tinka")
    plt.legend()
    plt.grid(True)
    
    # Show the probability
    plt.text(simulations * 0.7, min(win_percentages) + (max(win_percentages) - min(win_percentages)) * 0.05,
             f"Probability: {final_probability:.10f}",
             fontsize=12, color="red",
             ha="right", bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.show()


def analyze_number_frequency(df):
    """
    Analyzes the frequency of occurrence of each number in the historical Tinka results.

    Parameters:
        df (DataFrame): DataFrame containing the columns 'fecha' (date), 'bola' (ball position), and 'valor' (number).

    Returns:
        DataFrame: Table with each number and its frequency of occurrence.
    """

    freq = df[df["bola"] != "Boliyapa"]["valor"].value_counts().sort_index()
    freq_df = pd.DataFrame({"Number": freq.index, "Frequency": freq.values})

    plt.figure(figsize=(12, 5))

    # Estilo seaborn para mejorar estética
    sns.set_style("whitegrid")

    # Color plomo oscuro: #4B4B4B o similar
    # Borde negro en las barras
    bars = plt.bar(freq_df["Number"], freq_df["Frequency"],
                   color="#4B4B4B", edgecolor="black", linewidth=1.2)

    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("Frequency of number occurrences (without Boliyapa)")

    # Mostrar todos los números en el eje x
    plt.xticks(freq_df["Number"], rotation=0)

    # Mejorar visibilidad del grid solo en el eje y
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return freq_df


def hot_and_cold_numbers(df, recent_draws=10):
    """
    Finds hot and cold numbers in the most recent draws.
    
    Parameters:
        df (DataFrame): Historical Tinka results in long format.
        recent_draws (int): Number of recent draws to consider.
    
    Returns:
        tuple: (hot_numbers, cold_numbers)
    """
    
    recent_dates = df["fecha"].drop_duplicates().sort_values(ascending=False).head(recent_draws)
    recent_data = df[df["fecha"].isin(recent_dates)]
    
    freq_recent = recent_data[recent_data["bola"] != "Boliyapa"]["valor"].value_counts()
    hot_numbers = freq_recent.head(5).index.tolist()
    cold_numbers = freq_recent.tail(5).index.tolist()
    
    print(f"Hot numbers (last {recent_draws} draws): {hot_numbers}")
    print(f"Cold numbers (last {recent_draws} draws): {cold_numbers}")
    
    return hot_numbers, cold_numbers



def analyze_even_odd(df, last_n_games=None):
    """
    Analyzes how many even and odd numbers tend to appear in each draw.
    Shows two side-by-side bar charts: distribution of even counts and odd counts per draw,
    with data labels and colored borders.

    Parameters:
        df (DataFrame): Historical Tinka data with columns 'fecha', 'bola', 'valor'.
        last_n_games (int or None): If set, analyze only last N unique 'fecha' games.

    Returns:
        Tuple of Series: (even_counts_distribution, odd_counts_distribution)
    """

    # Filtrar fechas si se indica
    if last_n_games is not None:
        fechas_unicas = sorted(df['fecha'].unique())
        fechas_filtradas = fechas_unicas[-last_n_games:]
        df = df[df['fecha'].isin(fechas_filtradas)]

    # Excluir Boliyapa
    non_boliyapa = df[df["bola"] != "Boliyapa"]

    # Calcular número de pares e impares por fecha
    even_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n % 2 == 0 for n in x))
    odd_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n % 2 == 1 for n in x))

    # Distribuciones de frecuencia ordenadas
    dist_even = even_counts.value_counts().sort_index()
    dist_odd = odd_counts.value_counts().sort_index()

    sns.set_style("whitegrid")
    base_color = "#4B4B4B"  # plomo oscuro
    blue_edge = "#1F4E78"
    red_edge = "#C00000"

    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

    # Gráfico pares
    bars_even = sns.barplot(x=dist_even.index, y=dist_even.values, ax=axes[0],
                           color=base_color, edgecolor=blue_edge, linewidth=2)
    axes[0].set_title("Distribution of Even Numbers per Draw")
    axes[0].set_xlabel("Number of Even Numbers")
    axes[0].set_ylabel("Frequency of Draws")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas en barras pares
    for bar in bars_even.patches:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    # Gráfico impares
    bars_odd = sns.barplot(x=dist_odd.index, y=dist_odd.values, ax=axes[1],
                          color=base_color, edgecolor=red_edge, linewidth=2)
    axes[1].set_title("Distribution of Odd Numbers per Draw")
    axes[1].set_xlabel("Number of Odd Numbers")
    axes[1].set_ylabel("")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas en barras impares
    for bar in bars_odd.patches:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()




def analyze_low_high(df, split_point=25, last_n_games=None):
    """
    Analyzes how many low and high numbers tend to appear in each draw.
    Shows two side-by-side bar charts: distribution of low counts and high counts per draw,
    with data labels and colored borders.

    Parameters:
        df (DataFrame): Historical Tinka data with columns 'fecha', 'bola', 'valor'.
        split_point (int): Number dividing low and high numbers.
        last_n_games (int or None): If set, analyze only last N unique 'fecha' games.

    Returns:
        Tuple of Series: (low_counts_distribution, high_counts_distribution)
    """

    # Filtrar fechas si se indica
    if last_n_games is not None:
        fechas_unicas = sorted(df['fecha'].unique())
        fechas_filtradas = fechas_unicas[-last_n_games:]
        df = df[df['fecha'].isin(fechas_filtradas)]

    # Excluir Boliyapa
    non_boliyapa = df[df["bola"] != "Boliyapa"]

    # Calcular número de bajos y altos por fecha
    low_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n <= split_point for n in x))
    high_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n > split_point for n in x))

    # Distribuciones de frecuencia ordenadas
    dist_low = low_counts.value_counts().sort_index()
    dist_high = high_counts.value_counts().sort_index()

    sns.set_style("whitegrid")
    base_color = "#4B4B4B"  # plomo oscuro
    blue_edge = "#1f77b4"
    red_edge = "#d62728"

    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

    # Gráfico bajos
    bars_low = sns.barplot(x=dist_low.index, y=dist_low.values, ax=axes[0],
                           color=base_color, edgecolor=blue_edge, linewidth=2)
    axes[0].set_title(f"Distribution of Numbers ≤ {split_point} per Draw")
    axes[0].set_xlabel(f"Count of numbers ≤ {split_point}")
    axes[0].set_ylabel("Frequency of Draws")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas en barras bajos
    for bar in bars_low.patches:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    # Gráfico altos
    bars_high = sns.barplot(x=dist_high.index, y=dist_high.values, ax=axes[1],
                          color=base_color, edgecolor=red_edge, linewidth=2)
    axes[1].set_title(f"Distribution of Numbers > {split_point} per Draw")
    axes[1].set_xlabel(f"Count of numbers > {split_point}")
    axes[1].set_ylabel("")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas en barras altos
    for bar in bars_high.patches:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.1, f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def analyze_repeated_sequences(df_long):
    """
    Analyzes how many times exact combinations of 6 numbers repeat in the history.
    """
    combos = (df_long[df_long["bola"] != "Boliyapa"]
              .groupby("fecha")["valor"]
              .apply(lambda x: tuple(sorted(x))))
    
    combo_counts = combos.value_counts()
    repeated = combo_counts[combo_counts > 1]
    
    if repeated.empty:
        print("No repeated combinations in the history.")
    else:
        print("Repeated combinations:\n", repeated)



def analyze_consecutive_numbers(df_long, last_n_games=10):
    """
    Analyzes how many pairs of consecutive numbers appear in each draw,
    showing a horizontal bar plot with counts.

    Parameters:
        df_long (DataFrame): DataFrame con columnas 'fecha', 'bola', 'valor'.
        last_n_games (int): número de últimos sorteos a considerar (default 10).

    Returns:
        None: Muestra gráfico.
    """
    # Filtrar últimos N juegos por fecha
    fechas = sorted(df_long['fecha'].unique())
    fechas_filtradas = fechas[-last_n_games:]
    df_filtered = df_long[df_long['fecha'].isin(fechas_filtradas)]
    
    counts = []
    for _, group in df_filtered[df_filtered["bola"] != "Boliyapa"].groupby("fecha"):
        nums = sorted(group["valor"])
        consecutive_pairs = sum((b - a) == 1 for a, b in zip(nums, nums[1:]))
        counts.append(consecutive_pairs)
    
    # Contar frecuencia de ocurrencias
    freq = pd.Series(counts).value_counts().sort_index()

    # Configuración colores plomo y borde negro
    color_plomo = "#4B4B4B"
    edge_color = "black"
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bars = ax.barh(freq.index.astype(str), freq.values, color=color_plomo, edgecolor=edge_color, linewidth=1.5)

    # Etiquetas de dato (número encima de cada barra)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{int(width)}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Number of Consecutive Pairs")
    ax.set_title(f"Number of Consecutive Pairs per Draw (Last {last_n_games} Draws)")

    plt.tight_layout()
    plt.show()




def analyze_transitions(df_long, top_n=10, last_n_games=None):
    df_main = df_long[df_long["bola"] != "Boliyapa"].copy()
    df_main = df_main.sort_values("fecha")

    if last_n_games is not None:
        fechas = sorted(df_main['fecha'].unique())
        df_main = df_main[df_main['fecha'].isin(fechas[-last_n_games:])]

    draws = df_main.groupby("fecha")["valor"].apply(set).tolist()

    transitions = Counter()
    for prev, curr in zip(draws, draws[1:]):
        for p in prev:
            for c in curr:
                transitions[(p, c)] += 1

    df_trans = pd.DataFrame([
        {"prev": p, "curr": c, "count": cnt}
        for (p, c), cnt in transitions.items()
    ])

    top_pairs = df_trans.sort_values("count", ascending=False).head(top_n)
    top_pairs['transition'] = top_pairs.apply(lambda r: f"{r.prev}\u2192{r.curr}", axis=1)

    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=top_pairs, y='transition', x='count', color="#4B4B4B", edgecolor="black")

    # Agregar etiquetas de dato al final de cada barra
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.1, p.get_y() + p.get_height()/2,
                f'{int(width)}', va='center', ha='left', fontsize=10, fontweight='bold')

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Transition")
    ax.set_title(f"Top {top_n} Most Frequent Transitions Between Draws (Last {last_n_games if last_n_games else 'All'} Draws)")

    plt.tight_layout()
    plt.show()






def analyze_number_frequency_highlight(df_long, top_n=10, last_n_games=None):
    """
    Shows the frequency of numbers ordered from highest to lowest,
    highlighting the top N most frequent in blue and bottom N least frequent in red.
    Optionally, analyze only the last N games.

    Parameters:
        df_long (DataFrame): DataFrame with columns 'fecha', 'bola', 'valor'.
        top_n (int): Number of top and bottom numbers to highlight.
        last_n_games (int or None): If set, analyze only last N unique 'fecha' games.

    Returns:
        None: Displays the plot.
    """

    # Filtrar solo últimas fechas si se indica
    if last_n_games is not None:
        # Obtener las fechas únicas ordenadas de forma ascendente
        fechas_unicas = sorted(df_long['fecha'].unique())
        # Tomar las últimas fechas
        fechas_filtradas = fechas_unicas[-last_n_games:]
        df_filtered = df_long[df_long['fecha'].isin(fechas_filtradas)]
    else:
        df_filtered = df_long

    # Filtrar bolas distintas de 'Boliyapa'
    nums = df_filtered[df_filtered["bola"] != "Boliyapa"]["valor"]

    freq = nums.value_counts().sort_values(ascending=False)
    freq_df = pd.DataFrame({"Number": freq.index, "Frequency": freq.values})

    # Ordenar de mayor a menor explícitamente
    freq_df = freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

    # Colores
    base_color = "#4B4B4B"  # plomo oscuro
    top_color = "#1f77b4"   # azul profesional
    bottom_color = "#d62728"  # rojo profesional

    n = len(freq_df)
    colors = [top_color if i < top_n else bottom_color if i >= n - top_n else base_color for i in range(n)]

    plt.figure(figsize=(14, 5))
    sns.set_style("whitegrid")

    bars = plt.bar(range(n), freq_df["Frequency"], color=colors, edgecolor="black", linewidth=1)

    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title(f"Number Frequencies with Top {top_n} (blue) and Bottom {top_n} (red) Highlighted"
              + (f" in Last {last_n_games} Games" if last_n_games else ""))

    plt.xticks(ticks=range(n), labels=freq_df["Number"], rotation=0)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def analyze_repeats(df_tinka, last_n_games=None):
    """
    Analyzes how many winning numbers repeat from one draw to the next.

    Parameters:
        df_tinka (pd.DataFrame): DataFrame with columns ['fecha', 'bola', 'valor'].
        last_n_games (int or None): Number of last draws to consider.

    Returns:
        pd.DataFrame: Count and percentage of matches.
    """
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]

    # Filtrar últimos N juegos si aplica
    fechas = sorted(df_main['fecha'].unique())
    if last_n_games is not None:
        fechas = fechas[-last_n_games-1:]  # +1 porque comparamos pares consecutivos
    df_main = df_main[df_main['fecha'].isin(fechas)]

    draws = df_main.groupby('fecha')['valor'].apply(list).reset_index()
    draws = draws.sort_values('fecha').reset_index(drop=True)

    repeat_counts = []
    for i in range(1, len(draws)):
        prev_set = set(draws.loc[i-1, 'valor'])
        curr_set = set(draws.loc[i, 'valor'])
        matches = len(prev_set & curr_set)
        repeat_counts.append(matches)

    count_distribution = Counter(repeat_counts)

    df_counts = pd.DataFrame({
        'Matches': list(count_distribution.keys()),
        'Frequency': list(count_distribution.values())
    }).sort_values('Matches').reset_index(drop=True)

    total_compared = len(repeat_counts)
    df_counts['Percentage'] = (df_counts['Frequency'] / total_compared * 100).round(2)

    # Graficar
    plt.figure(figsize=(8, 5))
    color_plomo = "#4B4B4B"
    plt.bar(df_counts['Matches'], df_counts['Frequency'], color=color_plomo, edgecolor='black')
    plt.xlabel("Number of matches with the previous draw")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of matches between consecutive draws (Last {last_n_games if last_n_games else 'All'} draws)")
    plt.xticks(range(0, 7))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Etiquetas porcentaje arriba de barras
    for x, y, pct in zip(df_counts['Matches'], df_counts['Frequency'], df_counts['Percentage']):
        plt.text(x, y + 0.3, f"{pct}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()




def analyze_ball_position_ranges(df_tinka, range_step=10, last_n_games=None):
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    fechas = sorted(df_main['fecha'].unique())
    if last_n_games is not None:
        fechas = fechas[-last_n_games:]
    df_main = df_main[df_main['fecha'].isin(fechas)]

    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()

    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)

    stats_by_position = {}
    plt.figure(figsize=(14, 8))

    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        max_val = values_arr.max()
        bins = list(range(1, max_val + range_step, range_step))
        hist, bin_edges = np.histogram(values_arr, bins=bins)
        percentages = (hist / hist.sum() * 100).round(2)

        stats_df = pd.DataFrame({
            'Range': [f"{bin_edges[j]}-{bin_edges[j+1]-1}" for j in range(len(hist))],
            'Percentage': percentages
        })
        stats_by_position[ball_name] = stats_df

        plt.subplot(2, 3, i)
        sns.barplot(x='Range', y='Percentage', data=stats_df,
                    color="#4B4B4B", edgecolor="black")
        plt.title(f"{ball_name} - Distribution by ranges")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Distribution of each ball by position and range (Last {last_n_games if last_n_games else 'All'} draws)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def analyze_optimal_ranges_per_position(df_tinka, bins=5, last_n_games=None):
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    fechas = sorted(df_main['fecha'].unique())
    if last_n_games is not None:
        fechas = fechas[-last_n_games:]
    df_main = df_main[df_main['fecha'].isin(fechas)]

    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()

    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)

    stats_by_position = {}
    plt.figure(figsize=(14, 8))

    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        quantile_edges = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.percentile(values_arr, quantile_edges * 100).astype(int))
        edges[0] = 1
        edges[-1] = 50

        hist, bin_edges = np.histogram(values_arr, bins=edges)
        percentages = (hist / hist.sum() * 100).round(2)

        stats_df = pd.DataFrame({
            'Range': [f"{bin_edges[j]}-{bin_edges[j+1]}" for j in range(len(hist))],
            'Percentage': percentages
        })
        stats_by_position[ball_name] = stats_df

        plt.subplot(2, 3, i)
        sns.barplot(x='Range', y='Percentage', data=stats_df,
                    color="#4B4B4B", edgecolor="black")
        plt.title(f"{ball_name} - Optimal Ranges")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Distribution and Optimal Ranges by Position (Last {last_n_games if last_n_games else 'All'} draws)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_histograms_per_position(df_tinka, last_n_games=None):
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    fechas = sorted(df_main['fecha'].unique())
    if last_n_games is not None:
        fechas = fechas[-last_n_games:]
    df_main = df_main[df_main['fecha'].isin(fechas)]

    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()

    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)

    plt.figure(figsize=(14, 8))

    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        plt.subplot(2, 3, i)
        sns.histplot(values, bins=range(1, 52), color="#4B4B4B", edgecolor="black")
        plt.title(f"{ball_name} - Histogram")
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.xticks(range(1, 51, 2))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle(f"Historical Distribution by Position (Last {last_n_games if last_n_games else 'All'} draws)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
