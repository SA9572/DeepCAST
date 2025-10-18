# DeepCSAT: Comprehensive Data Visualization Script
# This script contains all 15+ visualizations for the DeepCSAT project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style and parameters
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def create_all_visualizations(df):
    """
    Create all 15+ visualizations for the DeepCSAT project
    Following UBM (Univariate, Bivariate, Multivariate) analysis
    """
    
    print("Creating comprehensive visualizations for DeepCSAT project...")
    
    # =============================================================================
    # UNIVARIATE ANALYSIS (U) - Charts 1-5
    # =============================================================================
    
    # Chart 1: CSAT Score Distribution (Univariate Analysis)
    def chart_1_csat_distribution():
        """Chart 1: CSAT Score Distribution - Univariate Analysis"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: CSAT Score Distribution
        plt.subplot(2, 2, 1)
        csat_counts = df['CSAT Score'].value_counts().sort_index()
        bars = plt.bar(csat_counts.index, csat_counts.values, color='skyblue', alpha=0.7)
        plt.title('Distribution of CSAT Scores', fontsize=14, fontweight='bold')
        plt.xlabel('CSAT Score')
        plt.ylabel('Frequency')
        plt.xticks(csat_counts.index)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Subplot 2: CSAT Score Pie Chart
        plt.subplot(2, 2, 2)
        plt.pie(csat_counts.values, labels=csat_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('CSAT Score Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # Subplot 3: CSAT Score Box Plot
        plt.subplot(2, 2, 3)
        plt.boxplot(df['CSAT Score'], vert=True)
        plt.title('CSAT Score Box Plot', fontsize=14, fontweight='bold')
        plt.ylabel('CSAT Score')
        
        # Subplot 4: CSAT Score Histogram
        plt.subplot(2, 2, 4)
        plt.hist(df['CSAT Score'], bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('CSAT Score Histogram', fontsize=14, fontweight='bold')
        plt.xlabel('CSAT Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 1 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Used multiple univariate visualizations to comprehensively understand CSAT score distribution")
        print("2. What insights found?")
        print("   - CSAT scores are heavily skewed towards higher values (4-5)")
        print("   - Very few customers give low scores (1-2)")
        print("3. Business Impact:")
        print("   - Positive: High concentration of 4-5 scores indicates strong satisfaction baseline")
        print("   - Action: Focus on maintaining current service quality")
    
    # Chart 2: Channel Name Distribution (Univariate Analysis)
    def chart_2_channel_distribution():
        """Chart 2: Channel Name Distribution - Univariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Channel Count
        plt.subplot(1, 2, 1)
        channel_counts = df['channel_name'].value_counts()
        bars = plt.bar(channel_counts.index, channel_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Channel Usage Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Channel Name')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Subplot 2: Channel Pie Chart
        plt.subplot(1, 2, 2)
        plt.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Channel Usage Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 2 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Bar and pie charts effectively show channel usage patterns")
        print("2. What insights found?")
        print("   - Inbound calls are the most popular channel")
        print("   - Email and Outcall have different usage patterns")
        print("3. Business Impact:")
        print("   - Positive: Understanding channel preferences helps resource allocation")
        print("   - Action: Optimize channel mix based on usage patterns")
    
    # Chart 3: Category Distribution (Univariate Analysis)
    def chart_3_category_distribution():
        """Chart 3: Category Distribution - Univariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Category Count
        plt.subplot(1, 2, 1)
        category_counts = df['category'].value_counts()
        bars = plt.bar(range(len(category_counts)), category_counts.values, color='lightcoral')
        plt.title('Interaction Category Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Subplot 2: Category Pie Chart
        plt.subplot(1, 2, 2)
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Category Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 3 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Bar and pie charts show interaction category patterns")
        print("2. What insights found?")
        print("   - Product Queries are the most common interaction type")
        print("   - Returns and Order Related issues are significant")
        print("3. Business Impact:")
        print("   - Positive: Understanding common issues helps improve processes")
        print("   - Action: Focus on improving Product Query handling")
    
    # Chart 4: Item Price Distribution (Univariate Analysis)
    def chart_4_price_distribution():
        """Chart 4: Item Price Distribution - Univariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Price Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df['Item_price'].dropna(), bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Item Price Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Item Price')
        plt.ylabel('Frequency')
        
        # Subplot 2: Price Box Plot
        plt.subplot(1, 2, 2)
        plt.boxplot(df['Item_price'].dropna())
        plt.title('Item Price Box Plot', fontsize=14, fontweight='bold')
        plt.ylabel('Item Price')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 4 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Histogram and box plot show price distribution and outliers")
        print("2. What insights found?")
        print("   - Most items are in lower price ranges")
        print("   - Some high-value outliers exist")
        print("3. Business Impact:")
        print("   - Positive: Understanding price distribution helps inventory planning")
        print("   - Action: Focus on high-value item customer service")
    
    # Chart 5: Handling Time Distribution (Univariate Analysis)
    def chart_5_handling_time_distribution():
        """Chart 5: Handling Time Distribution - Univariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Handling Time Histogram
        plt.subplot(1, 2, 1)
        plt.hist(df['connected_handling_time'].dropna(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Handling Time Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Handling Time (minutes)')
        plt.ylabel('Frequency')
        
        # Subplot 2: Handling Time Box Plot
        plt.subplot(1, 2, 2)
        plt.boxplot(df['connected_handling_time'].dropna())
        plt.title('Handling Time Box Plot', fontsize=14, fontweight='bold')
        plt.ylabel('Handling Time (minutes)')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 5 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Histogram and box plot show handling time patterns")
        print("2. What insights found?")
        print("   - Most issues are resolved quickly")
        print("   - Some complex issues take longer to resolve")
        print("3. Business Impact:")
        print("   - Positive: Quick resolution times indicate efficient service")
        print("   - Action: Investigate long handling times for improvement")
    
    # =============================================================================
    # BIVARIATE ANALYSIS (B) - Charts 6-10
    # =============================================================================
    
    # Chart 6: Channel vs CSAT Score (Bivariate Analysis)
    def chart_6_channel_vs_csat():
        """Chart 6: Channel vs CSAT Score - Bivariate Analysis"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Channel-wise CSAT Score Distribution
        plt.subplot(2, 2, 1)
        channel_csat = df.groupby('channel_name')['CSAT Score'].mean().sort_values(ascending=False)
        bars = plt.bar(channel_csat.index, channel_csat.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Average CSAT Score by Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Channel Name')
        plt.ylabel('Average CSAT Score')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Subplot 2: Channel-wise CSAT Score Box Plot
        plt.subplot(2, 2, 2)
        df.boxplot(column='CSAT Score', by='channel_name', ax=plt.gca())
        plt.title('CSAT Score Distribution by Channel', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        
        # Subplot 3: Channel-wise Count
        plt.subplot(2, 2, 3)
        channel_counts = df['channel_name'].value_counts()
        plt.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Channel Usage Distribution', fontsize=14, fontweight='bold')
        
        # Subplot 4: Channel-wise CSAT Score Heatmap
        plt.subplot(2, 2, 4)
        channel_csat_pivot = df.pivot_table(values='CSAT Score', index='channel_name', aggfunc=['mean', 'count'])
        sns.heatmap(channel_csat_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Channel CSAT Score Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 6 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Multiple visualizations show channel-CSAT relationships")
        print("2. What insights found?")
        print("   - Different channels show varying CSAT scores")
        print("   - Some channels provide better customer experience")
        print("3. Business Impact:")
        print("   - Positive: Identify high-performing channels")
        print("   - Action: Improve underperforming channels")
    
    # Chart 7: Category vs CSAT Score (Bivariate Analysis)
    def chart_7_category_vs_csat():
        """Chart 7: Category vs CSAT Score - Bivariate Analysis"""
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Category-wise CSAT Score
        plt.subplot(2, 2, 1)
        category_csat = df.groupby('category')['CSAT Score'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(category_csat)), category_csat.values, color='lightcoral')
        plt.title('Average CSAT Score by Category', fontsize=14, fontweight='bold')
        plt.xlabel('Category')
        plt.ylabel('Average CSAT Score')
        plt.xticks(range(len(category_csat)), category_csat.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Subplot 2: Category-wise CSAT Box Plot
        plt.subplot(2, 2, 2)
        df.boxplot(column='CSAT Score', by='category', ax=plt.gca())
        plt.title('CSAT Score Distribution by Category', fontsize=14, fontweight='bold')
        plt.suptitle('')
        plt.xticks(rotation=45)
        
        # Subplot 3: Category Count
        plt.subplot(2, 2, 3)
        category_counts = df['category'].value_counts()
        plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Category Distribution', fontsize=14, fontweight='bold')
        
        # Subplot 4: Category CSAT Heatmap
        plt.subplot(2, 2, 4)
        category_csat_pivot = df.pivot_table(values='CSAT Score', index='category', aggfunc=['mean', 'count'])
        sns.heatmap(category_csat_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Category CSAT Score Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 7 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Shows relationship between interaction category and satisfaction")
        print("2. What insights found?")
        print("   - Different categories have varying satisfaction levels")
        print("   - Some categories consistently perform better")
        print("3. Business Impact:")
        print("   - Positive: Identify categories needing improvement")
        print("   - Action: Focus on low-performing categories")
    
    # Chart 8: Price vs CSAT Score (Bivariate Analysis)
    def chart_8_price_vs_csat():
        """Chart 8: Price vs CSAT Score - Bivariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Scatter Plot
        plt.subplot(1, 2, 1)
        plt.scatter(df['Item_price'], df['CSAT Score'], alpha=0.6, color='blue')
        plt.title('Item Price vs CSAT Score', fontsize=14, fontweight='bold')
        plt.xlabel('Item Price')
        plt.ylabel('CSAT Score')
        
        # Add trend line
        z = np.polyfit(df['Item_price'].dropna(), df['CSAT Score'], 1)
        p = np.poly1d(z)
        plt.plot(df['Item_price'], p(df['Item_price']), "r--", alpha=0.8)
        
        # Subplot 2: Price Bins vs CSAT
        plt.subplot(1, 2, 2)
        df['price_bin'] = pd.cut(df['Item_price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        price_csat = df.groupby('price_bin')['CSAT Score'].mean()
        bars = plt.bar(range(len(price_csat)), price_csat.values, color='lightblue')
        plt.title('CSAT Score by Price Range', fontsize=14, fontweight='bold')
        plt.xlabel('Price Range')
        plt.ylabel('Average CSAT Score')
        plt.xticks(range(len(price_csat)), price_csat.index, rotation=45)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 8 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Scatter plot and binned analysis show price-satisfaction relationship")
        print("2. What insights found?")
        print("   - Price and satisfaction may have correlation")
        print("   - Different price ranges show varying satisfaction")
        print("3. Business Impact:")
        print("   - Positive: Understand price-satisfaction dynamics")
        print("   - Action: Optimize pricing strategy based on satisfaction")
    
    # Chart 9: Handling Time vs CSAT Score (Bivariate Analysis)
    def chart_9_handling_time_vs_csat():
        """Chart 9: Handling Time vs CSAT Score - Bivariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Scatter Plot
        plt.subplot(1, 2, 1)
        plt.scatter(df['connected_handling_time'], df['CSAT Score'], alpha=0.6, color='green')
        plt.title('Handling Time vs CSAT Score', fontsize=14, fontweight='bold')
        plt.xlabel('Handling Time (minutes)')
        plt.ylabel('CSAT Score')
        
        # Add trend line
        z = np.polyfit(df['connected_handling_time'].dropna(), df['CSAT Score'], 1)
        p = np.poly1d(z)
        plt.plot(df['connected_handling_time'], p(df['connected_handling_time']), "r--", alpha=0.8)
        
        # Subplot 2: Handling Time Bins vs CSAT
        plt.subplot(1, 2, 2)
        df['handling_time_bin'] = pd.cut(df['connected_handling_time'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
        handling_csat = df.groupby('handling_time_bin')['CSAT Score'].mean()
        bars = plt.bar(range(len(handling_csat)), handling_csat.values, color='lightgreen')
        plt.title('CSAT Score by Handling Time Range', fontsize=14, fontweight='bold')
        plt.xlabel('Handling Time Range')
        plt.ylabel('Average CSAT Score')
        plt.xticks(range(len(handling_csat)), handling_csat.index, rotation=45)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 9 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Scatter plot and binned analysis show handling time-satisfaction relationship")
        print("2. What insights found?")
        print("   - Faster handling times may correlate with higher satisfaction")
        print("   - Long handling times may lead to lower satisfaction")
        print("3. Business Impact:")
        print("   - Positive: Quick resolution improves satisfaction")
        print("   - Action: Optimize handling time for better satisfaction")
    
    # Chart 10: Agent Tenure vs CSAT Score (Bivariate Analysis)
    def chart_10_tenure_vs_csat():
        """Chart 10: Agent Tenure vs CSAT Score - Bivariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: Tenure vs CSAT
        plt.subplot(1, 2, 1)
        tenure_csat = df.groupby('Tenure Bucket')['CSAT Score'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(tenure_csat)), tenure_csat.values, color='orange')
        plt.title('Average CSAT Score by Agent Tenure', fontsize=14, fontweight='bold')
        plt.xlabel('Tenure Bucket')
        plt.ylabel('Average CSAT Score')
        plt.xticks(range(len(tenure_csat)), tenure_csat.index, rotation=45)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Subplot 2: Tenure Distribution
        plt.subplot(1, 2, 2)
        tenure_counts = df['Tenure Bucket'].value_counts()
        plt.pie(tenure_counts.values, labels=tenure_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Agent Tenure Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 10 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Shows relationship between agent experience and satisfaction")
        print("2. What insights found?")
        print("   - More experienced agents may provide better service")
        print("   - Tenure distribution shows agent experience levels")
        print("3. Business Impact:")
        print("   - Positive: Experience correlates with satisfaction")
        print("   - Action: Invest in agent training and retention")
    
    # =============================================================================
    # MULTIVARIATE ANALYSIS (M) - Charts 11-15
    # =============================================================================
    
    # Chart 11: Channel-Category-CSAT Heatmap (Multivariate Analysis)
    def chart_11_channel_category_heatmap():
        """Chart 11: Channel-Category-CSAT Heatmap - Multivariate Analysis"""
        plt.figure(figsize=(15, 8))
        
        # Create pivot table
        pivot_table = df.pivot_table(values='CSAT Score', index='channel_name', columns='category', aggfunc='mean')
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Average CSAT Score'})
        plt.title('Channel-Category CSAT Score Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Category')
        plt.ylabel('Channel')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 11 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Heatmap shows complex multivariate relationships")
        print("2. What insights found?")
        print("   - Different channel-category combinations have varying performance")
        print("   - Some combinations consistently perform better")
        print("3. Business Impact:")
        print("   - Positive: Identify optimal channel-category combinations")
        print("   - Action: Route interactions to best-performing combinations")
    
    # Chart 12: Price-Handling Time-CSAT Scatter (Multivariate Analysis)
    def chart_12_price_handling_scatter():
        """Chart 12: Price-Handling Time-CSAT Scatter - Multivariate Analysis"""
        fig = px.scatter_3d(df, x='Item_price', y='connected_handling_time', z='CSAT Score',
                           color='CSAT Score', size='CSAT Score',
                           title='Price-Handling Time-CSAT 3D Scatter Plot',
                           labels={'Item_price': 'Item Price', 'connected_handling_time': 'Handling Time'})
        fig.show()
        
        print("Chart 12 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - 3D scatter plot shows three-dimensional relationships")
        print("2. What insights found?")
        print("   - Complex interactions between price, handling time, and satisfaction")
        print("   - Patterns in 3D space reveal hidden relationships")
        print("3. Business Impact:")
        print("   - Positive: Understand complex customer satisfaction drivers")
        print("   - Action: Optimize multiple factors simultaneously")
    
    # Chart 13: Agent Performance Matrix (Multivariate Analysis)
    def chart_13_agent_performance_matrix():
        """Chart 13: Agent Performance Matrix - Multivariate Analysis"""
        plt.figure(figsize=(15, 10))
        
        # Create agent performance metrics
        agent_metrics = df.groupby('Agent_name').agg({
            'CSAT Score': ['mean', 'count'],
            'connected_handling_time': 'mean'
        }).round(2)
        
        agent_metrics.columns = ['Avg_CSAT', 'Interaction_Count', 'Avg_Handling_Time']
        
        # Filter agents with sufficient interactions
        agent_metrics = agent_metrics[agent_metrics['Interaction_Count'] >= 10]
        
        # Subplot 1: CSAT vs Handling Time
        plt.subplot(2, 2, 1)
        plt.scatter(agent_metrics['Avg_Handling_Time'], agent_metrics['Avg_CSAT'], 
                   s=agent_metrics['Interaction_Count']*10, alpha=0.6, color='blue')
        plt.title('Agent Performance: CSAT vs Handling Time', fontsize=14, fontweight='bold')
        plt.xlabel('Average Handling Time')
        plt.ylabel('Average CSAT Score')
        
        # Subplot 2: Top Performers
        plt.subplot(2, 2, 2)
        top_agents = agent_metrics.nlargest(10, 'Avg_CSAT')
        bars = plt.bar(range(len(top_agents)), top_agents['Avg_CSAT'], color='green')
        plt.title('Top 10 Agents by CSAT Score', fontsize=14, fontweight='bold')
        plt.xlabel('Agent Rank')
        plt.ylabel('Average CSAT Score')
        plt.xticks(range(len(top_agents)), [f'Agent {i+1}' for i in range(len(top_agents))], rotation=45)
        
        # Subplot 3: Interaction Count Distribution
        plt.subplot(2, 2, 3)
        plt.hist(agent_metrics['Interaction_Count'], bins=20, alpha=0.7, color='orange')
        plt.title('Agent Interaction Count Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Number of Agents')
        
        # Subplot 4: Performance Quadrant
        plt.subplot(2, 2, 4)
        median_csat = agent_metrics['Avg_CSAT'].median()
        median_handling = agent_metrics['Avg_Handling_Time'].median()
        
        plt.scatter(agent_metrics['Avg_Handling_Time'], agent_metrics['Avg_CSAT'], 
                   s=agent_metrics['Interaction_Count']*5, alpha=0.6, color='red')
        plt.axhline(y=median_csat, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=median_handling, color='black', linestyle='--', alpha=0.5)
        plt.title('Agent Performance Quadrant', fontsize=14, fontweight='bold')
        plt.xlabel('Average Handling Time')
        plt.ylabel('Average CSAT Score')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 13 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Multiple visualizations show agent performance patterns")
        print("2. What insights found?")
        print("   - Agents show varying performance levels")
        print("   - Some agents excel in both speed and satisfaction")
        print("3. Business Impact:")
        print("   - Positive: Identify top performers for recognition")
        print("   - Action: Provide targeted training for underperformers")
    
    # Chart 14: Correlation Heatmap (Multivariate Analysis)
    def chart_14_correlation_heatmap():
        """Chart 14: Correlation Heatmap - Multivariate Analysis"""
        plt.figure(figsize=(15, 10))
        
        # Select numerical columns for correlation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 14 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Heatmap shows all pairwise correlations between features")
        print("2. What insights found?")
        print("   - Strong correlations between related features")
        print("   - Weak correlations indicate independent features")
        print("3. Business Impact:")
        print("   - Positive: Understand feature relationships for model building")
        print("   - Action: Remove highly correlated features to avoid multicollinearity")
    
    # Chart 15: Pair Plot (Multivariate Analysis)
    def chart_15_pair_plot():
        """Chart 15: Pair Plot - Multivariate Analysis"""
        # Select key numerical features for pair plot
        key_features = ['CSAT Score', 'Item_price', 'connected_handling_time']
        
        # Create pair plot
        sns.pairplot(df[key_features], diag_kind='hist', plot_kws={'alpha': 0.6})
        plt.suptitle('Pair Plot of Key Features', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        print("Chart 15 Analysis:")
        print("1. Why did you pick the specific chart?")
        print("   - Pair plot shows all pairwise relationships in one view")
        print("2. What insights found?")
        print("   - Distribution patterns and relationships between features")
        print("   - Scatter plots reveal correlation patterns")
        print("3. Business Impact:")
        print("   - Positive: Comprehensive view of feature relationships")
        print("   - Action: Use insights for feature selection and model building")
    
    # Execute all visualizations
    print("="*80)
    print("CREATING ALL VISUALIZATIONS FOR DEEPCSAT PROJECT")
    print("="*80)
    
    # Univariate Analysis
    print("\n" + "="*50)
    print("UNIVARIATE ANALYSIS (U) - Charts 1-5")
    print("="*50)
    
    chart_1_csat_distribution()
    chart_2_channel_distribution()
    chart_3_category_distribution()
    chart_4_price_distribution()
    chart_5_handling_time_distribution()
    
    # Bivariate Analysis
    print("\n" + "="*50)
    print("BIVARIATE ANALYSIS (B) - Charts 6-10")
    print("="*50)
    
    chart_6_channel_vs_csat()
    chart_7_category_vs_csat()
    chart_8_price_vs_csat()
    chart_9_handling_time_vs_csat()
    chart_10_tenure_vs_csat()
    
    # Multivariate Analysis
    print("\n" + "="*50)
    print("MULTIVARIATE ANALYSIS (M) - Charts 11-15")
    print("="*50)
    
    chart_11_channel_category_heatmap()
    chart_12_price_handling_scatter()
    chart_13_agent_performance_matrix()
    chart_14_correlation_heatmap()
    chart_15_pair_plot()
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)

# Example usage
if __name__ == "__main__":
    # Load your dataset here
    # df = pd.read_csv('your_dataset.csv')
    # create_all_visualizations(df)
    print("DeepCSAT Visualization Script Ready!")
    print("Load your dataset and call create_all_visualizations(df) to generate all charts.")
