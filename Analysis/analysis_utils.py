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
    
    plt.figure(figsize=(10, 4))
    plt.bar(freq.index, freq.values)
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("Frequency of number occurrences (without Boliyapa)")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.show()
    
    return pd.DataFrame({"Number": freq.index, "Frequency": freq.values})

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

def analyze_even_odd(df):
    """
    Analyzes how many even and odd numbers tend to appear in each draw.
    
    Parameters:
        df (DataFrame): Historical Tinka data.
    
    Returns:
        Series: Count of occurrences by number of evens.
    """
    non_boliyapa = df[df["bola"] != "Boliyapa"]
    even_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n % 2 == 0 for n in x))
    
    dist = even_counts.value_counts().sort_index()
    plt.bar(dist.index, dist.values)
    plt.xlabel("Number of even numbers")
    plt.ylabel("Frequency of draws")
    plt.title("Distribution of evens in draws")
    plt.show()
    
    return dist

def analyze_low_high(df, split_point=25):
    """
    Analyzes how many low and high numbers tend to appear in each draw.
    
    Parameters:
        df (DataFrame): Historical Tinka data.
        split_point (int): Number that divides the low/high range.
    
    Returns:
        Series: Count of occurrences by number of lows.
    """
    non_boliyapa = df[df["bola"] != "Boliyapa"]
    low_counts = non_boliyapa.groupby("fecha")["valor"].apply(lambda x: sum(n <= split_point for n in x))
    
    dist = low_counts.value_counts().sort_index()
    plt.bar(dist.index, dist.values)
    plt.xlabel(f"Number of numbers ≤ {split_point}")
    plt.ylabel("Frequency of draws")
    plt.title("Distribution of low numbers in draws")
    plt.show()
    
    return dist

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

def analyze_consecutive_numbers(df_long):
    """
    Analyzes how many pairs of consecutive numbers appear in each draw.
    """
    counts = []
    
    for _, group in df_long[df_long["bola"] != "Boliyapa"].groupby("fecha"):
        nums = sorted(group["valor"])
        consecutive_pairs = sum((b - a) == 1 for a, b in zip(nums, nums[1:]))
        counts.append(consecutive_pairs)
    
    plt.figure(figsize=(6,4))
    sns.countplot(x=counts, palette="viridis")
    plt.title("Number of consecutive pairs per draw")
    plt.xlabel("Consecutive pairs")
    plt.ylabel("Frequency")
    plt.show()

def analyze_transitions(df_long, top_n=10):
    """
    Shows a simple transition matrix: frequency with which a number appears
    after another in the next draw.
    """
    df_main = df_long[df_long["bola"] != "Boliyapa"].copy()
    df_main = df_main.sort_values("fecha")
    
    # Group by date
    draws = df_main.groupby("fecha")["valor"].apply(set).tolist()
    
    transitions = Counter()
    for prev, curr in zip(draws, draws[1:]):
        for p in prev:
            for c in curr:
                transitions[(p, c)] += 1
    
    # Convert to DataFrame for display
    df_trans = pd.DataFrame([
        {"prev": p, "curr": c, "count": cnt}
        for (p, c), cnt in transitions.items()
    ])
    
    top_pairs = df_trans.sort_values("count", ascending=False).head(top_n)
    
    plt.figure(figsize=(8,5))
    sns.barplot(data=top_pairs, x="count", y=top_pairs.apply(lambda r: f"{r.prev}→{r.curr}", axis=1), palette="magma")
    plt.title(f"Top {top_n} most frequent transitions between draws")
    plt.xlabel("Frequency")
    plt.ylabel("Transition")
    plt.show()


def analyze_rare_numbers(df_long, top_n=10):
    """
    Shows the least frequent numbers in history.
    """
    nums = df_long[df_long["bola"] != "Boliyapa"]["valor"]
    freq = nums.value_counts().sort_values().head(top_n)
    
    plt.figure(figsize=(8,4))
    sns.barplot(x=freq.index, y=freq.values, palette="coolwarm")
    plt.title(f"Top {top_n} least frequent numbers")
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.show()


def analyze_repeats(df_tinka):
    """
    Analyzes how many winning numbers repeat from one draw to the next.
    
    Parameters:
        df_tinka (pd.DataFrame): DataFrame with columns ['fecha', 'bola', 'valor'].
                                 Must contain the historical data of Tinka.
                                 
    Returns:
        pd.DataFrame: Count and percentage of matches.
    """
    # Filter only the 6 main balls (without Boliyapa)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Group by date and create lists of drawn numbers
    draws = df_main.groupby('fecha')['valor'].apply(list).reset_index()
    draws = draws.sort_values('fecha').reset_index(drop=True)
    
    repeat_counts = []

    # Compare each draw with the previous one
    for i in range(1, len(draws)):
        prev_set = set(draws.loc[i-1, 'valor'])
        curr_set = set(draws.loc[i, 'valor'])
        matches = len(prev_set & curr_set)
        repeat_counts.append(matches)
    
    # Count occurrences
    count_distribution = Counter(repeat_counts)
    
    # Convert to DataFrame for display
    df_counts = pd.DataFrame({
        'Matches': list(count_distribution.keys()),
        'Frequency': list(count_distribution.values())
    }).sort_values('Matches').reset_index(drop=True)
    
    # Calculate percentage
    total_compared = len(repeat_counts)
    df_counts['Percentage'] = (df_counts['Frequency'] / total_compared * 100).round(2)
    
    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.bar(df_counts['Matches'], df_counts['Frequency'], color='skyblue', edgecolor='black')
    plt.xlabel("Number of matches with the previous draw")
    plt.ylabel("Frequency")
    plt.title("Distribution of matches between consecutive draws")
    
    # Show percentages on the bars
    for i, (x, y, pct) in enumerate(zip(df_counts['Matches'], df_counts['Frequency'], df_counts['Percentage'])):
        plt.text(x, y + 0.5, f"{pct}%", ha='center', fontsize=9)
    
    plt.xticks(range(0, 7))  # Only values 0 to 6
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    


def analyze_ball_position_ranges(df_tinka, range_step=10):
    """
    Analyzes the distribution of each ball by its position (sorted from lowest to highest)
    and shows percentages of occurrence by ranges.
    
    Parameters:
        df_tinka (pd.DataFrame): DataFrame with columns ['fecha', 'bola', 'valor'].
                                 Must contain the historical data of Tinka.
        range_step (int): Size of the range to group values (default 10).
    
    Returns:
        dict: Dictionary with percentage statistics by range for each position.
    """
    # Filter only main balls (B1 to B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Group by date and sort each draw
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    # Create structure to store ranges
    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    
    # Fill with values by position
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)
    
    stats_by_position = {}
    
    plt.figure(figsize=(14, 8))
    
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        
        # Calculate ranges
        max_val = values_arr.max()
        bins = list(range(1, max_val + range_step, range_step))
        hist, bin_edges = np.histogram(values_arr, bins=bins)
        
        # Calculate percentages
        percentages = (hist / hist.sum() * 100).round(2)
        
        # Save in dictionary
        stats_df = pd.DataFrame({
            'Range': [f"{bin_edges[j]}-{bin_edges[j+1]-1}" for j in range(len(hist))],
            'Frequency': hist,
            'Percentage': percentages
        })
        stats_by_position[ball_name] = stats_df
        
        # Plot
        plt.subplot(2, 3, i)
        plt.bar(stats_df['Range'], stats_df['Percentage'], color='skyblue', edgecolor='black')
        plt.title(f"{ball_name} - Distribution by ranges")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Distribution of each ball by position and range", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def analyze_optimal_ranges_per_position(df_tinka, bins=5):
    """
    Analyzes the distribution of each ball by position (sorted) and determines optimal ranges.
    
    Parameters:
        df_tinka (pd.DataFrame): DataFrame with columns ['fecha', 'bola', 'valor'].
        bins (int): Number of ranges to create per position (quantiles).
    
    Returns:
        dict: Optimal ranges by position with percentages.
    """
    # Filter only main balls (B1 to B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Group by date and sort each draw
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    
    # Fill lists for each position
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)
    
    stats_by_position = {}
    
    plt.figure(figsize=(14, 8))
    
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        values_arr = np.array(values)
        
        # Calculate quantiles for optimal cuts
        quantile_edges = np.linspace(0, 1, bins+1)
        edges = np.unique(np.percentile(values_arr, quantile_edges * 100).astype(int))
        
        # Ensure that the edges cover everything
        edges[0] = 1
        edges[-1] = 50
        
        hist, bin_edges = np.histogram(values_arr, bins=edges)
        percentages = (hist / hist.sum() * 100).round(2)
        
        stats_df = pd.DataFrame({
            'Range': [f"{bin_edges[j]}-{bin_edges[j+1]}" for j in range(len(hist))],
            'Frequency': hist,
            'Percentage': percentages
        })
        stats_by_position[ball_name] = stats_df
        
        # Plot
        plt.subplot(2, 3, i)
        plt.bar(stats_df['Range'], stats_df['Percentage'], color='lightgreen', edgecolor='black')
        plt.title(f"{ball_name} - Optimal Ranges")
        plt.xticks(rotation=45)
        plt.ylabel("%")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Distribution and Optimal Ranges by Position", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_histograms_per_position(df_tinka):
    """
    Displays independent histograms for each ball by position.
    
    Parameters:
        df_tinka (pd.DataFrame): DataFrame with columns ['fecha', 'bola', 'valor'].
    
    Returns:
        dict: List of values by position.
    """
    # Filter only main balls (B1 to B6)
    df_main = df_tinka[df_tinka['bola'].str.startswith('B') & (df_tinka['bola'] != 'Boliyapa')]
    
    # Group by date and sort each draw
    draws_sorted = df_main.groupby('fecha')['valor'].apply(lambda x: sorted(x)).reset_index()
    
    ball_positions = {f"Ball {i+1}": [] for i in range(6)}
    
    # Fill lists for each position
    for _, row in draws_sorted.iterrows():
        for i, val in enumerate(row['valor']):
            ball_positions[f"Ball {i+1}"].append(val)
    
    # Plot histograms
    plt.figure(figsize=(14, 8))
    for i, (ball_name, values) in enumerate(ball_positions.items(), start=1):
        plt.subplot(2, 3, i)
        plt.hist(values, bins=range(1, 52), color='skyblue', edgecolor='black', align='left')
        plt.title(f"{ball_name} - Histogram")
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.xticks(range(1, 51, 2))  # Numbers from 1 to 50 every 2 to avoid crowding
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Historical Distribution by Position (Ordered Balls)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


