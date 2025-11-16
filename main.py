"""
NFL Running Back Metrics Analysis
Analyzes correlations between Speed Score, Rushing Efficiency, and performance metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load the NFL running back data"""
    data = {
        'Player': ['Jonathan Taylor', 'James Cook', 'Rico Dowdle', 'J.K. Dobbins', 'Javonte Williams',
                   'Bijan Robinson', 'Devon Achane', 'Derrick Henry', 'Travis Etienne', 'Breece Hall',
                   'Christian McCaffrey', 'Kyren Williams', 'Jahmyr Gibbs', 'Ashton Jeanty', 'Josh Jacobs',
                   'Quinshon Judkins', "D'Andre Swift", 'Saquon Barkley', 'Tony Pollard', 'Kenneth Walker',
                   'Jacory Croskey-Merritt', 'Chase Brown', 'Jordan Mason', 'Cam Skattebo', 'Jaylen Warren',
                   'Alvin Kamara', 'David Montgomery', 'Kyle Monangai', 'Nick Chubb', 'Isiah Pacheco',
                   'Omarion Hampton', 'Kimani Vidal', 'Chuba Hubbard', 'TreVeyon Henderson', 'Rachaad White',
                   'Kareem Hunt', 'Rhamondre Stevenson', 'Woody Marks', 'Tyler Allgeier', 'Blake Corum',
                   'Bucky Irving', 'Zach Charbonnet', 'Kenneth Gainwell', 'R.J. Harvey', 'Tyrone Tracy'],
        'Rushing_Efficiency': [3.06, 3.1, 2.85, 3.46, 3.07, 3.93, 3.87, 3.52, 3.43, 3.49,
                              4.62, 3.52, 3.97, 4.15, 3.88, 3.84, 3.7, 3.98, 3.86, 4.6,
                              3.36, 3.82, 4.24, 3.74, 3.97, 4.29, 3.62, 3.37, 3.62, 3.7,
                              3.7, 3.51, 3.88, 4.19, 3.92, 3.2, 4.5, 4.07, 4.78, 3.91,
                              4.78, 5.55, 4.29, 4.92, 4.74],
        'Speed_Score': [121.7, 104.28, 101.22, 109.1, 100.88, 108.67, 109.68, 118.63, 112.06, 116.85,
                       100.29, 89.98, 110.14, 107.61, 98.27, 109.73, 105.26, 124.33, 100.62, 114.66,
                       105.06, 109.05, 101.36, 93.68, 100.33, 98.98, 96.62, 94.2, 108.77, 118.46,
                       111.7, 107.66, 116.23, 104.9, 106.25, 95.65, 99.67, 97.44, 100.05, 97.36,
                       89.6, 101.64, 104.38, 109.4, 103.77],
        'Avg_YPC': [6, 5.6, 5.5, 5, 5.2, 5, 5, 4.8, 5, 5,
                    3.5, 4.4, 4.9, 3.8, 3.8, 4, 4.9, 4.1, 4, 4.5,
                    4.7, 3.9, 4.2, 4.1, 4.1, 3.5, 4.4, 5.2, 3.9, 4.2,
                    4.8, 4.7, 3.6, 4.2, 3.7, 4, 3.4, 3.7, 3.6, 4.6,
                    3.3, 2.9, 4.2, 4.3, 3.2],
        'Avg_TD_per_Game': [1.5, 0.778, 0.5, 0.4, 0.889, 0.222, 0.5, 0.667, 0.333, 0.222,
                           0.4, 0.556, 0.889, 0.625, 1.222, 0.444, 0.5, 0.444, 0.222, 0.333,
                           0.25, 0.444, 0.1, 0.444, 0.3, 0.222, 0.222, 0.125, 0.286, 0.222,
                           0.444, 0.125, 0.125, 0.625, 0.222, 0.111, 0.375, 0.125, 0.111, 0.125,
                           0.125, 0.75, 0.375, 0.258, 0.111],
        'Games_Played': [10, 9, 10, 10, 9, 9, 10, 9, 9, 9,
                        10, 9, 9, 8, 9, 9, 8, 9, 9, 9,
                        8, 9, 10, 9, 10, 9, 9, 8, 7, 9,
                        9, 8, 8, 8, 9, 9, 8, 8, 9, 8,
                        8, 8, 8, 8, 9],
        'Rushing_TD': [15, 7, 5, 4, 8, 2, 5, 6, 3, 2,
                      4, 5, 8, 5, 11, 4, 4, 4, 2, 3,
                      2, 4, 1, 4, 3, 2, 2, 1, 2, 2,
                      4, 1, 1, 5, 2, 1, 3, 1, 1, 1,
                      1, 6, 3, 2, 1]
    }
    
    return pd.DataFrame(data)

# Load data for interactive use
df = load_data()

def create_correlation_matrix(df):
    """Create and display correlation matrix heatmap"""
    # Select numeric columns for correlation
    numeric_cols = ['Rushing_Efficiency', 'Speed_Score', 'Avg_YPC', 'Avg_TD_per_Game', 'Rushing_TD']
    correlation_matrix = df[numeric_cols].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    plt.title('Correlation Matrix: NFL Running Back Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_scatter_plots(df):
    """Create scatter plots with trend lines for key relationships"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot 1: Speed Score vs Rushing Efficiency
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df['Speed_Score'], df['Rushing_Efficiency'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    z1 = np.polyfit(df['Speed_Score'], df['Rushing_Efficiency'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(df['Speed_Score'], p1(df['Speed_Score']), "r--", alpha=0.8, label=f'Trend: y={z1[0]:.4f}x+{z1[1]:.2f}')
    
    # Calculate and display correlation
    corr1, pval1 = stats.pearsonr(df['Speed_Score'], df['Rushing_Efficiency'])
    ax1.text(0.05, 0.95, f'r = {corr1:.3f}\np = {pval1:.3f}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Speed Score', fontsize=11)
    ax1.set_ylabel('Rushing Efficiency', fontsize=11)
    ax1.set_title('Speed Score vs Rushing Efficiency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: Speed Score vs Avg TD per Game
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['Speed_Score'], df['Avg_TD_per_Game'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    z2 = np.polyfit(df['Speed_Score'], df['Avg_TD_per_Game'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['Speed_Score'], p2(df['Speed_Score']), "r--", alpha=0.8, label=f'Trend: y={z2[0]:.4f}x+{z2[1]:.2f}')
    
    corr2, pval2 = stats.pearsonr(df['Speed_Score'], df['Avg_TD_per_Game'])
    ax2.text(0.05, 0.95, f'r = {corr2:.3f}\np = {pval2:.3f}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Speed Score', fontsize=11)
    ax2.set_ylabel('Avg TD per Game', fontsize=11)
    ax2.set_title('Speed Score vs Avg TD per Game', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Rushing Efficiency vs Avg TD per Game
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df['Rushing_Efficiency'], df['Avg_TD_per_Game'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    z3 = np.polyfit(df['Rushing_Efficiency'], df['Avg_TD_per_Game'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(df['Rushing_Efficiency'], p3(df['Rushing_Efficiency']), "r--", alpha=0.8, label=f'Trend: y={z3[0]:.4f}x+{z3[1]:.2f}')
    
    corr3, pval3 = stats.pearsonr(df['Rushing_Efficiency'], df['Avg_TD_per_Game'])
    ax3.text(0.05, 0.95, f'r = {corr3:.3f}\np = {pval3:.3f}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlabel('Rushing Efficiency', fontsize=11)
    ax3.set_ylabel('Avg TD per Game', fontsize=11)
    ax3.set_title('Rushing Efficiency vs Avg TD per Game', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: Speed Score vs Avg YPC
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['Speed_Score'], df['Avg_YPC'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    z4 = np.polyfit(df['Speed_Score'], df['Avg_YPC'], 1)
    p4 = np.poly1d(z4)
    ax4.plot(df['Speed_Score'], p4(df['Speed_Score']), "r--", alpha=0.8, label=f'Trend: y={z4[0]:.4f}x+{z4[1]:.2f}')
    
    corr4, pval4 = stats.pearsonr(df['Speed_Score'], df['Avg_YPC'])
    ax4.text(0.05, 0.95, f'r = {corr4:.3f}\np = {pval4:.3f}', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_xlabel('Speed Score', fontsize=11)
    ax4.set_ylabel('Avg Yards per Carry', fontsize=11)
    ax4.set_title('Speed Score vs Avg YPC', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Plot 5: Rushing Efficiency vs Avg YPC
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(df['Rushing_Efficiency'], df['Avg_YPC'], alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    z5 = np.polyfit(df['Rushing_Efficiency'], df['Avg_YPC'], 1)
    p5 = np.poly1d(z5)
    ax5.plot(df['Rushing_Efficiency'], p5(df['Rushing_Efficiency']), "r--", alpha=0.8, label=f'Trend: y={z5[0]:.4f}x+{z5[1]:.2f}')
    
    corr5, pval5 = stats.pearsonr(df['Rushing_Efficiency'], df['Avg_YPC'])
    ax5.text(0.05, 0.95, f'r = {corr5:.3f}\np = {pval5:.3f}', 
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax5.set_xlabel('Rushing Efficiency', fontsize=11)
    ax5.set_ylabel('Avg Yards per Carry', fontsize=11)
    ax5.set_title('Rushing Efficiency vs Avg YPC', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # Plot 6: 3D-like visualization with bubble size
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(df['Speed_Score'], df['Rushing_Efficiency'], 
                         s=df['Avg_TD_per_Game']*300, # Size based on TD rate
                         c=df['Avg_YPC'], # Color based on YPC
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Avg YPC', fontsize=10)
    
    ax6.set_xlabel('Speed Score', fontsize=11)
    ax6.set_ylabel('Rushing Efficiency', fontsize=11)
    ax6.set_title('Multi-Dimensional View\n(Size = TD Rate, Color = YPC)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('NFL Running Back Performance Correlations', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_distribution_analysis(df):
    """Create distribution plots for key metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Speed Score distribution
    axes[0, 0].hist(df['Speed_Score'], bins=15, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['Speed_Score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Speed_Score"].mean():.1f}')
    axes[0, 0].axvline(df['Speed_Score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Speed_Score"].median():.1f}')
    axes[0, 0].set_xlabel('Speed Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Speed Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rushing Efficiency distribution
    axes[0, 1].hist(df['Rushing_Efficiency'], bins=15, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['Rushing_Efficiency'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Rushing_Efficiency"].mean():.2f}')
    axes[0, 1].axvline(df['Rushing_Efficiency'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Rushing_Efficiency"].median():.2f}')
    axes[0, 1].set_xlabel('Rushing Efficiency')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Rushing Efficiency Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Avg TD per Game distribution
    axes[0, 2].hist(df['Avg_TD_per_Game'], bins=15, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(df['Avg_TD_per_Game'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Avg_TD_per_Game"].mean():.3f}')
    axes[0, 2].axvline(df['Avg_TD_per_Game'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Avg_TD_per_Game"].median():.3f}')
    axes[0, 2].set_xlabel('Avg TD per Game')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Avg TD per Game Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Box plots
    axes[1, 0].boxplot([df['Speed_Score'], df['Rushing_Efficiency']*20, df['Avg_TD_per_Game']*100],
                      labels=['Speed Score', 'Rush Eff (x20)', 'TD/Game (x100)'])
    axes[1, 0].set_title('Comparative Box Plots (Scaled)')
    axes[1, 0].set_ylabel('Scaled Values')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Violin plot for Speed Score by performance tier
    df['Performance_Tier'] = pd.qcut(df['Avg_TD_per_Game'], q=3, labels=['Low', 'Medium', 'High'])
    parts = axes[1, 1].violinplot([df[df['Performance_Tier'] == tier]['Speed_Score'].values 
                                   for tier in ['Low', 'Medium', 'High']],
                                  positions=[1, 2, 3], widths=0.6, showmeans=True, showmedians=True)
    axes[1, 1].set_xticks([1, 2, 3])
    axes[1, 1].set_xticklabels(['Low TD Rate', 'Medium TD Rate', 'High TD Rate'])
    axes[1, 1].set_ylabel('Speed Score')
    axes[1, 1].set_title('Speed Score by TD Performance Tier')
    axes[1, 1].grid(True, alpha=0.3)
    
    # QQ plot for normality check
    stats.probplot(df['Speed_Score'], dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Speed Score Q-Q Plot (Normality Check)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution Analysis of Running Back Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_performance_ranking(df):
    """Create a comprehensive performance ranking visualization"""
    # Calculate composite score (weighted average)
    df['Composite_Score'] = (
        df['Speed_Score'] / df['Speed_Score'].max() * 0.3 +
        (1 / df['Rushing_Efficiency']) / (1 / df['Rushing_Efficiency']).max() * 0.3 +  # Lower is better for efficiency
        df['Avg_TD_per_Game'] / df['Avg_TD_per_Game'].max() * 0.4
    )
    
    # Sort by composite score
    df_sorted = df.sort_values('Composite_Score', ascending=False).head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart of top performers
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))
    bars = ax1.barh(df_sorted['Player'], df_sorted['Composite_Score'], color=colors, edgecolor='black')
    ax1.set_xlabel('Composite Performance Score', fontsize=12)
    ax1.set_title('Top 15 Running Backs - Composite Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Radar chart for top 5 performers
    top_5 = df_sorted.head(5)
    
    categories = ['Speed Score\n(Normalized)', 'Rush Efficiency\n(Inverted)', 'TD Rate\n(Normalized)', 
                 'YPC\n(Normalized)', 'Total TDs\n(Normalized)']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax2 = plt.subplot(122, projection='polar')
    
    for idx, player in enumerate(top_5['Player'].values):
        player_data = top_5.iloc[idx]
        values = [
            player_data['Speed_Score'] / df['Speed_Score'].max(),
            1 - (player_data['Rushing_Efficiency'] - df['Rushing_Efficiency'].min()) / 
                (df['Rushing_Efficiency'].max() - df['Rushing_Efficiency'].min()),
            player_data['Avg_TD_per_Game'] / df['Avg_TD_per_Game'].max(),
            player_data['Avg_YPC'] / df['Avg_YPC'].max(),
            player_data['Rushing_TD'] / df['Rushing_TD'].max()
        ]
        values += values[:1]
        
        ax2.plot(angles, values, 'o-', linewidth=2, label=player, markersize=6)
        ax2.fill(angles, values, alpha=0.15)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Top 5 RBs - Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax2.grid(True)
    
    plt.suptitle('NFL Running Back Performance Rankings', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def print_statistical_summary(df):
    """Print statistical summary and key findings"""
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY - NFL RUNNING BACK ANALYSIS")
    print("="*60)
    
    # Correlation analysis
    print("\nKEY CORRELATIONS:")
    print("-"*40)
    
    correlations = {
        'Speed Score vs Rush Efficiency': stats.pearsonr(df['Speed_Score'], df['Rushing_Efficiency']),
        'Speed Score vs TD Rate': stats.pearsonr(df['Speed_Score'], df['Avg_TD_per_Game']),
        'Rush Efficiency vs TD Rate': stats.pearsonr(df['Rushing_Efficiency'], df['Avg_TD_per_Game']),
        'Speed Score vs YPC': stats.pearsonr(df['Speed_Score'], df['Avg_YPC']),
        'Rush Efficiency vs YPC': stats.pearsonr(df['Rushing_Efficiency'], df['Avg_YPC'])
    }
    
    for name, (corr, pval) in correlations.items():
        significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"{name:30} r = {corr:6.3f}, p = {pval:6.4f} {significance}")
    
    print("\n" + "-"*40)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    
    # Descriptive statistics
    print("\nDESCRIPTIVE STATISTICS:")
    print("-"*40)
    print(df[['Speed_Score', 'Rushing_Efficiency', 'Avg_YPC', 'Avg_TD_per_Game']].describe())
    
    # Top performers
    print("\nTOP PERFORMERS BY METRIC:")
    print("-"*40)
    
    print(f"\nSpeed Score Leaders:")
    for idx, row in df.nlargest(3, 'Speed_Score')[['Player', 'Speed_Score']].iterrows():
        print(f"  {row['Player']:25} {row['Speed_Score']:6.2f}")
    
    print(f"\nRushing Efficiency Leaders (Lower is Better):")
    for idx, row in df.nsmallest(3, 'Rushing_Efficiency')[['Player', 'Rushing_Efficiency']].iterrows():
        print(f"  {row['Player']:25} {row['Rushing_Efficiency']:6.2f}")
    
    print(f"\nTD Rate Leaders:")
    for idx, row in df.nlargest(3, 'Avg_TD_per_Game')[['Player', 'Avg_TD_per_Game']].iterrows():
        print(f"  {row['Player']:25} {row['Avg_TD_per_Game']:6.3f}")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Calculate and interpret the main finding
    ss_re_corr = stats.pearsonr(df['Speed_Score'], df['Rushing_Efficiency'])[0]
    ss_td_corr = stats.pearsonr(df['Speed_Score'], df['Avg_TD_per_Game'])[0]
    
    print(f"""
1. Speed Score vs Rushing Efficiency: {'NEGATIVE' if ss_re_corr < 0 else 'POSITIVE'} correlation (r={ss_re_corr:.3f})
   - {"Higher speed scores correlate with LOWER (better) rushing efficiency" if ss_re_corr < 0 else "Higher speed scores correlate with HIGHER (worse) rushing efficiency"}
   
2. Speed Score vs TD Production: {'POSITIVE' if ss_td_corr > 0 else 'NEGATIVE'} correlation (r={ss_td_corr:.3f})
   - {"Bigger, faster players tend to score more TDs" if ss_td_corr > 0 else "Speed Score shows little relationship with TD production"}
   
3. The data suggests that while explosive players (high Speed Score) may not always
   be the most efficient, they can still be valuable for TD production.
   
4. Players like Jonathan Taylor and Saquon Barkley combine high Speed Scores
   with strong performance metrics, representing ideal RB profiles.
""")

def main():
    """Main execution function"""
    # Load data
    df = load_data()
    print("Data loaded successfully!")
    print(f"Analyzing {len(df)} NFL running backs...")
    
    # Print statistical summary
    print_statistical_summary(df)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    # 1. Correlation Matrix
    fig1 = create_correlation_matrix(df)
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation matrix saved as 'correlation_matrix.png'")
    
    # 2. Scatter Plots
    fig2 = create_scatter_plots(df)
    plt.savefig('scatter_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Scatter plots saved as 'scatter_analysis.png'")
    
    # 3. Distribution Analysis
    fig3 = create_distribution_analysis(df)
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Distribution analysis saved as 'distribution_analysis.png'")
    
    # 4. Performance Rankings
    fig4 = create_performance_ranking(df)
    plt.savefig('performance_rankings.png', dpi=300, bbox_inches='tight')
    print("✓ Performance rankings saved as 'performance_rankings.png'")
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*60)
    print("Analysis complete! All visualizations have been generated.")
    print("="*60)

if __name__ == "__main__":
    main()