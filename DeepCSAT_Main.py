# DeepCSAT: Main Execution Script
# Complete pipeline for E-Commerce Customer Satisfaction Score Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from DeepCSAT_ML_Models import DeepCSATMLPipeline
from DeepCSAT_NLP_Processing import DeepCSATNLPProcessor
from DeepCSAT_Visualizations import create_all_visualizations
from DeepCSAT_Deployment import DeepCSATDeployment

class DeepCSATMain:
    """
    Main execution class for DeepCSAT project
    Orchestrates the complete pipeline from data loading to deployment
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.ml_pipeline = None
        self.nlp_processor = None
        self.deployment = None
        self.results = {}
        
    def load_data(self, data_path=None):
        """Load the dataset"""
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            # Try to find the dataset
            import os
            possible_paths = [
                'DEEP-CSAT Project-20251009T024510Z-1-001/DEEP-CSAT Project/eCommerce_Customer_support_data.csv',
                'data/eCommerce_Customer_support_data.csv',
                'eCommerce_Customer_support_data.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    break
        
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError("Dataset not found. Please provide the correct path.")
        
        print(f"Loading dataset from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        
        return self.df
    
    def run_complete_analysis(self):
        """Run the complete DeepCSAT analysis pipeline"""
        print("="*80)
        print("DEEPCSAT: E-COMMERCE CUSTOMER SATISFACTION SCORE PREDICTION")
        print("="*80)
        
        # Step 1: Data Loading and Initial Exploration
        print("\n" + "="*50)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("="*50)
        
        if self.df is None:
            self.load_data()
        
        # Basic data exploration
        self._explore_data()
        
        # Step 2: Data Preprocessing and Feature Engineering
        print("\n" + "="*50)
        print("STEP 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
        print("="*50)
        
        self._preprocess_data()
        
        # Step 3: NLP Processing
        print("\n" + "="*50)
        print("STEP 3: NLP PROCESSING")
        print("="*50)
        
        self._process_nlp()
        
        # Step 4: Data Visualization
        print("\n" + "="*50)
        print("STEP 4: DATA VISUALIZATION")
        print("="*50)
        
        self._create_visualizations()
        
        # Step 5: Machine Learning Models
        print("\n" + "="*50)
        print("STEP 5: MACHINE LEARNING MODELS")
        print("="*50)
        
        self._train_models()
        
        # Step 6: Model Evaluation
        print("\n" + "="*50)
        print("STEP 6: MODEL EVALUATION")
        print("="*50)
        
        self._evaluate_models()
        
        # Step 7: Hypothesis Testing
        print("\n" + "="*50)
        print("STEP 7: HYPOTHESIS TESTING")
        print("="*50)
        
        self._perform_hypothesis_testing()
        
        # Step 8: Deployment Preparation
        print("\n" + "="*50)
        print("STEP 8: DEPLOYMENT PREPARATION")
        print("="*50)
        
        self._prepare_deployment()
        
        print("\n" + "="*80)
        print("DEEPCSAT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return self.results
    
    def _explore_data(self):
        """Perform initial data exploration"""
        print("Performing initial data exploration...")
        
        # Basic information
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        
        # Missing values
        missing_values = self.df.isnull().sum()
        print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
        
        # Target variable analysis
        if 'CSAT Score' in self.df.columns:
            print(f"\nCSAT Score distribution:")
            print(self.df['CSAT Score'].value_counts().sort_index())
            print(f"CSAT Score statistics:\n{self.df['CSAT Score'].describe()}")
        
        # Save basic statistics
        self.results['data_info'] = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': missing_values.to_dict(),
            'csat_distribution': self.df['CSAT Score'].value_counts().to_dict() if 'CSAT Score' in self.df.columns else None
        }
    
    def _preprocess_data(self):
        """Preprocess the data"""
        print("Preprocessing data...")
        
        # Initialize ML pipeline for preprocessing
        self.ml_pipeline = DeepCSATMLPipeline(self.df)
        self.ml_pipeline.data_preprocessing()
        
        # Update dataframe with processed data
        self.df = self.ml_pipeline.df
        
        print("Data preprocessing completed!")
    
    def _process_nlp(self):
        """Process NLP features"""
        print("Processing NLP features...")
        
        # Initialize NLP processor
        self.nlp_processor = DeepCSATNLPProcessor(self.df)
        
        # Preprocess text
        self.nlp_processor.preprocess_text('Customer Remarks')
        
        # Extract text features
        self.nlp_processor.extract_text_features()
        
        # Analyze sentiment
        self.nlp_processor.analyze_sentiment()
        
        # Perform topic modeling
        self.nlp_processor.perform_topic_modeling(n_topics=5)
        
        # Update dataframe with NLP features
        self.df = self.nlp_processor.df
        
        print("NLP processing completed!")
    
    def _create_visualizations(self):
        """Create all visualizations"""
        print("Creating comprehensive visualizations...")
        
        # Create all 15+ visualizations
        create_all_visualizations(self.df)
        
        print("Visualizations completed!")
    
    def _train_models(self):
        """Train machine learning models"""
        print("Training machine learning models...")
        
        # Train all models
        self.ml_pipeline.train_models()
        
        # Save models
        self.ml_pipeline.save_models()
        
        print("Model training completed!")
    
    def _evaluate_models(self):
        """Evaluate and compare models"""
        print("Evaluating models...")
        
        # Evaluate models
        results_df = self.ml_pipeline.evaluate_models()
        
        # Save results
        self.results['model_performance'] = results_df.to_dict()
        
        print("Model evaluation completed!")
    
    def _perform_hypothesis_testing(self):
        """Perform statistical hypothesis testing"""
        print("Performing hypothesis testing...")
        
        # Hypothesis 1: Channel type affects CSAT scores
        self._test_channel_csat_hypothesis()
        
        # Hypothesis 2: Handling time affects CSAT scores
        self._test_handling_time_csat_hypothesis()
        
        # Hypothesis 3: Agent experience affects CSAT scores
        self._test_agent_experience_hypothesis()
        
        print("Hypothesis testing completed!")
    
    def _test_channel_csat_hypothesis(self):
        """Test hypothesis about channel type and CSAT scores"""
        from scipy.stats import f_oneway
        
        print("Testing Hypothesis 1: Channel type affects CSAT scores")
        
        # Group CSAT scores by channel
        channel_groups = [group['CSAT Score'].values for name, group in self.df.groupby('channel_name')]
        
        # Perform ANOVA test
        f_stat, p_value = f_oneway(*channel_groups)
        
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Result: {'Reject null hypothesis' if p_value < 0.05 else 'Fail to reject null hypothesis'}")
        
        self.results['hypothesis_1'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _test_handling_time_csat_hypothesis(self):
        """Test hypothesis about handling time and CSAT scores"""
        from scipy.stats import pearsonr
        
        print("Testing Hypothesis 2: Handling time affects CSAT scores")
        
        # Calculate correlation
        correlation, p_value = pearsonr(
            self.df['connected_handling_time'].dropna(),
            self.df['CSAT Score']
        )
        
        print(f"Correlation coefficient: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Result: {'Significant correlation' if p_value < 0.05 else 'No significant correlation'}")
        
        self.results['hypothesis_2'] = {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _test_agent_experience_hypothesis(self):
        """Test hypothesis about agent experience and CSAT scores"""
        from scipy.stats import f_oneway
        
        print("Testing Hypothesis 3: Agent experience affects CSAT scores")
        
        # Group CSAT scores by tenure bucket
        tenure_groups = [group['CSAT Score'].values for name, group in self.df.groupby('Tenure Bucket')]
        
        # Perform ANOVA test
        f_stat, p_value = f_oneway(*tenure_groups)
        
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Result: {'Reject null hypothesis' if p_value < 0.05 else 'Fail to reject null hypothesis'}")
        
        self.results['hypothesis_3'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _prepare_deployment(self):
        """Prepare deployment environment"""
        print("Preparing deployment environment...")
        
        # Initialize deployment system
        self.deployment = DeepCSATDeployment()
        
        # Load models
        self.deployment.load_models()
        
        # Setup production environment
        self.deployment.setup_production_environment()
        
        # Generate documentation
        self.deployment.generate_documentation()
        
        print("Deployment preparation completed!")
    
    def save_results(self, filepath='results.json'):
        """Save analysis results"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive analysis report...")
        
        report = f"""
# DeepCSAT Analysis Report

## Executive Summary
This report presents the results of the DeepCSAT project for predicting customer satisfaction scores in e-commerce platforms.

## Dataset Overview
- **Total Records**: {self.results['data_info']['shape'][0]:,}
- **Features**: {self.results['data_info']['shape'][1]}
- **Target Variable**: CSAT Score (1-5 scale)

## Key Findings

### 1. Data Quality
- Missing values handled successfully
- Data preprocessing completed with feature engineering
- NLP features extracted from customer remarks

### 2. Model Performance
"""
        
        if 'model_performance' in self.results:
            report += "| Model | R² Score | MAE | RMSE |\n"
            report += "|-------|----------|-----|------|\n"
            
            for model, metrics in self.results['model_performance'].items():
                if isinstance(metrics, dict) and 'test_r2' in metrics:
                    report += f"| {model} | {metrics['test_r2']:.4f} | {metrics['test_mae']:.4f} | {metrics['test_rmse']:.4f} |\n"
        
        report += """
### 3. Statistical Analysis
"""
        
        if 'hypothesis_1' in self.results:
            report += f"- **Channel Type Impact**: {'Significant' if self.results['hypothesis_1']['significant'] else 'Not significant'}\n"
        
        if 'hypothesis_2' in self.results:
            report += f"- **Handling Time Impact**: {'Significant' if self.results['hypothesis_2']['significant'] else 'Not significant'}\n"
        
        if 'hypothesis_3' in self.results:
            report += f"- **Agent Experience Impact**: {'Significant' if self.results['hypothesis_3']['significant'] else 'Not significant'}\n"
        
        report += """
## Business Recommendations

1. **Focus on High-Performing Channels**: Allocate more resources to channels with higher CSAT scores
2. **Optimize Handling Time**: Reduce response times to improve customer satisfaction
3. **Agent Training**: Invest in training programs for agents with lower experience
4. **Real-time Monitoring**: Implement the deployed model for real-time CSAT prediction

## Technical Implementation

- **Models Trained**: Multiple ML models including Deep Learning ANN
- **Deployment Ready**: Production-ready API with Docker containerization
- **Monitoring**: Prometheus-based monitoring system
- **Documentation**: Comprehensive API documentation

## Conclusion

The DeepCSAT project successfully demonstrates the application of machine learning and deep learning techniques for customer satisfaction prediction in e-commerce platforms. The deployed system provides real-time insights and actionable recommendations for improving customer experience.
"""
        
        with open('DeepCSAT_Report.md', 'w') as f:
            f.write(report)
        
        print("Analysis report generated: DeepCSAT_Report.md")

def main():
    """Main execution function"""
    print("DeepCSAT: E-Commerce Customer Satisfaction Score Prediction")
    print("="*60)
    
    # Initialize the main system
    deepcsat = DeepCSATMain()
    
    try:
        # Run complete analysis
        results = deepcsat.run_complete_analysis()
        
        # Save results
        deepcsat.save_results()
        
        # Generate report
        deepcsat.generate_report()
        
        print("\n" + "="*60)
        print("DEEPCSAT PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review the generated report: DeepCSAT_Report.md")
        print("2. Check the results: results.json")
        print("3. Deploy the system: ./deploy.sh")
        print("4. Test the API: python test_api.py")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
