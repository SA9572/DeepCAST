# DeepCSAT Setup Script
# Automated setup for the DeepCSAT project

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        'data',
        'models',
        'config',
        'results',
        'logs',
        'monitoring',
        'visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_dataset():
    """Check if dataset exists"""
    print("Checking for dataset...")
    
    possible_paths = [
        'DEEP-CSAT Project-20251009T024510Z-1-001/DEEP-CSAT Project/eCommerce_Customer_support_data.csv',
        'data/eCommerce_Customer_support_data.csv',
        'eCommerce_Customer_support_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found at: {path}")
            return True
    
    print("❌ Dataset not found. Please ensure the dataset is available.")
    print("Expected locations:")
    for path in possible_paths:
        print(f"  - {path}")
    
    return False

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    modules = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'tensorflow',
        'nltk',
        'textblob',
        'plotly'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All modules imported successfully")
        return True

def create_sample_data():
    """Create sample data for testing if dataset is not available"""
    print("Creating sample data for testing...")
    
    import pandas as pd
    import numpy as np
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    
    sample_data = {
        'Unique id': [f'sample_{i}' for i in range(n_samples)],
        'channel_name': np.random.choice(['Inbound', 'Outcall', 'Email'], n_samples),
        'category': np.random.choice(['Product Queries', 'Order Related', 'Returns', 'Cancellation'], n_samples),
        'Sub-category': np.random.choice(['Product Specific Information', 'Installation/demo', 'Exchange / Replacement', 'Not Needed'], n_samples),
        'Customer Remarks': [f'Sample remark {i}' for i in range(n_samples)],
        'Order_id': [f'order_{i}' for i in range(n_samples)],
        'order_date_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Issue_reported at': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'issue_responded': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Survey_response_Date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Customer_City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'Product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples),
        'Item_price': np.random.uniform(10, 1000, n_samples),
        'connected_handling_time': np.random.uniform(1, 60, n_samples),
        'Agent_name': [f'Agent_{i}' for i in range(n_samples)],
        'Supervisor': [f'Supervisor_{i}' for i in range(n_samples)],
        'Manager': [f'Manager_{i}' for i in range(n_samples)],
        'Tenure Bucket': np.random.choice(['0-30', '30-90', '>90', 'On Job Training'], n_samples),
        'Agent Shift': np.random.choice(['Morning', 'Evening', 'Split'], n_samples),
        'CSAT Score': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.35, 0.35])
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to data directory
    df.to_csv('data/sample_data.csv', index=False)
    
    print("✅ Sample data created: data/sample_data.csv")
    return True

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("Running quick test...")
    
    try:
        # Test data loading
        import pandas as pd
        df = pd.read_csv('data/sample_data.csv')
        print(f"✅ Data loaded: {df.shape}")
        
        # Test basic analysis
        print(f"✅ CSAT distribution: {df['CSAT Score'].value_counts().to_dict()}")
        
        # Test visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        df['CSAT Score'].value_counts().plot(kind='bar')
        plt.title('CSAT Score Distribution')
        plt.savefig('visualizations/test_plot.png')
        plt.close()
        print("✅ Visualization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("DEEPCSAT PROJECT SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check for dataset
    dataset_available = check_dataset()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during requirements installation")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed during NLTK data download")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed during import testing")
        sys.exit(1)
    
    # Create sample data if dataset not available
    if not dataset_available:
        create_sample_data()
    
    # Run quick test
    if not run_quick_test():
        print("❌ Setup failed during quick test")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ DEEPCSAT SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\nNext Steps:")
    print("1. Run the complete analysis:")
    print("   python DeepCSAT_Main.py")
    print("\n2. Or use Jupyter notebook:")
    print("   jupyter notebook DeepCSAT_Project.ipynb")
    print("\n3. For deployment:")
    print("   python DeepCSAT_Deployment.py")
    
    print("\nProject Structure:")
    print("- DeepCSAT_Main.py: Main execution script")
    print("- DeepCSAT_Project.ipynb: Jupyter notebook")
    print("- DeepCSAT_ML_Models.py: ML pipeline")
    print("- DeepCSAT_NLP_Processing.py: NLP processing")
    print("- DeepCSAT_Visualizations.py: Visualization functions")
    print("- DeepCSAT_Deployment.py: Deployment system")
    
    print("\nDocumentation:")
    print("- README.md: Project overview")
    print("- PROJECT_STRUCTURE.md: Detailed structure")
    print("- API_DOCUMENTATION.md: API documentation (after deployment)")

if __name__ == "__main__":
    main()
