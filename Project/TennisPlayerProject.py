import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_columns', None)
df = pd.read_csv("C:/Users/sebas/Downloads/CombinedDF.csv")



def minutesVSw_svpt():
    # Scatter plot
    plt.scatter(df['minutes'], df['w_aces'])
    plt.title('Scatter Plot of Minutes vs. W_Svpt')
    plt.xlabel('Minutes')
    plt.ylabel('W_Svpt')
    plt.show()

def surfaceVSacesBoxplot():
    df.boxplot(by='surface', column=['l_ace'], grid=True)
    plt.title('Boxplot of Aces')
    plt.xlabel('Surfaces')
    plt.ylabel('Aces')
    plt.show()

#surfaceVSacesBoxplot()

def acesVSheightANDsurface():
    #plt.figure(figsize=(20, 12))

    # Scatter plot with points colored by surface
    #sns.scatterplot(x='winner_ht', y='w_ace', hue='surface', data=df, alpha=0.5)

    # Regression lines for each surface type
    sns.lmplot(x='winner_ht', y='w_ace', hue='surface', data=df, ci=None, fit_reg=True, scatter=True, height=15)

    plt.title('Scatter Plot of Winner Height vs. W_Ace with Regression Lines for Each Surface')
    plt.xlabel('Winner Height')
    plt.ylabel('W_Ace')
    plt.show()


def loserDfVSwinnerDf():
    plt.figure(figsize=(10, 6))

    # Scatter plot with points colored by surface
    #sns.scatterplot(x='winner_ht', y='w_df', hue='surface', data=df, alpha=0.5)
    sns.lmplot(x='winner_ht', y='w_df', data=df, ci=None, fit_reg=True, scatter=True, height=7, hue='surface')
    plt.title('Scatter Plot of Winner Height vs. W_Ace with Regression Lines for Each Surface')
    plt.xlabel('Winner Height')
    plt.ylabel('Winner Double Faults')
    plt.show()


def winnerVSloserBpSaved():
    # Calculate the difference between winners and losers for the 'bpSaved' column
    df['bpSaved_diff'] = df['l_bpSaved'] - df['w_bpSaved']

    # Group by the 'round' column and calculate the mean of the difference
    round_diff_mean = df.groupby('round')['bpSaved_diff'].mean()

    # Plot the bar chart
    plt.bar(round_diff_mean.index, round_diff_mean)
    plt.xlabel('Round')
    plt.ylabel('Mean Difference in BpSaved (Losers - Winners)')
    plt.title('Discrepancy in BpSaved between Winners and Losers by Round')
    plt.show()

#winnerVSloserBpSaved()

def winnerVSloserBpFaced():
    df['bpSaved_diff'] = df['l_bpFaced'] - df['w_bpFaced']

    # Group by the 'round' column and calculate the mean of the difference
    round_diff_mean = df.groupby('round')['bpSaved_diff'].mean()

    # Plot the bar chart
    plt.bar(round_diff_mean.index, round_diff_mean)
    plt.xlabel('Round')
    plt.ylabel('Mean Difference in BpFaced (Losers - Winners)')
    plt.title('Discrepancy in BpFaced between Winners and Losers by Round')
    plt.show()

#winnerVSloserBpFaced()

def durationVscount():
    fig, ax = plt.subplots()
    duration_counts = df.groupby('minutes').size().reset_index(name='match_count')
    ax.bar(duration_counts['minutes'], duration_counts['match_count'], label='Match Count', alpha=0.7)
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Match Count')
    ax.set_title('Relationship Between Match Count and Duration')
    ax.set_xlim(0, 400)
    ax.legend()
    plt.show()
    print(df)

def durVSco():
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')
    fig, ax = plt.subplots()
    duration_surface_counts = df.groupby(['minutes', 'surface']).size().reset_index(name='match_count')
    unique_surfaces = df['surface'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_surfaces)))
    for i, surface in enumerate(unique_surfaces):
        surface_data = duration_surface_counts[duration_surface_counts['surface'] == surface]
        ax.bar(surface_data['minutes'], surface_data['match_count'], label=surface, color=colors[i], alpha=0.7)
    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Match Count')
    ax.set_title('Relationship Between Match Count and Duration')
    ax.set_xlim(0, 400)
    ax.legend()
    plt.show()

#durVSco()

def ageVScount():
    df['age_difference'] = df['winner_age'] - df['loser_age']

    age_difference_counts = df.groupby(['round', 'age_difference']).size().reset_index(name='match_count')

    colormap = plt.cm.get_cmap('tab10', len(age_difference_counts['round'].unique()))
    for rnd in age_difference_counts['round'].unique():
        round_data = age_difference_counts[age_difference_counts['round'] == rnd]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(round_data['age_difference'], round_data['match_count'], label=rnd, color=colormap(0), alpha=0.7)
        ax.set_xlabel('Age Difference (Winner - Loser)')
        ax.set_ylabel(f'Match Count - {rnd}')
        ax.set_title(f'Relationship Between Age Difference and Match Count - {rnd}')
        ax.legend()
        plt.show()


ageVScount()