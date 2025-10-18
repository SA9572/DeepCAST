# GitHub Repository Setup Script for DeepCSAT Project
# This script helps set up and upload the DeepCSAT project to GitHub

import os
import subprocess
import sys
import json
from pathlib import Path

class GitHubSetup:
    def __init__(self, repo_name="DeepCSAT", username=None):
        self.repo_name = repo_name
        self.username = username
        self.repo_url = f"https://github.com/{username}/{repo_name}.git" if username else None
        
    def check_git_config(self):
        """Check if git is configured"""
        try:
            # Check git user name
            result = subprocess.run(['git', 'config', '--global', 'user.name'], 
                                  capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                print("❌ Git user name not configured")
                return False
            
            # Check git user email
            result = subprocess.run(['git', 'config', '--global', 'user.email'], 
                                  capture_output=True, text=True)
            if result.returncode != 0 or not result.stdout.strip():
                print("❌ Git user email not configured")
                return False
            
            print("✅ Git configuration found")
            return True
            
        except FileNotFoundError:
            print("❌ Git not found. Please install Git first.")
            return False
    
    def configure_git(self):
        """Configure git if not already configured"""
        print("Configuring Git...")
        
        # Get user input for git configuration
        name = input("Enter your Git username: ").strip()
        email = input("Enter your Git email: ").strip()
        
        try:
            subprocess.run(['git', 'config', '--global', 'user.name', name], check=True)
            subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
            print("✅ Git configured successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to configure Git")
            return False
    
    def initialize_git_repo(self):
        """Initialize git repository"""
        print("Initializing Git repository...")
        
        try:
            # Initialize git repository
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
            
            # Add all files
            subprocess.run(['git', 'add', '.'], check=True)
            print("✅ Files added to staging")
            
            # Create initial commit
            subprocess.run(['git', 'commit', '-m', 'Initial commit: DeepCSAT project setup'], check=True)
            print("✅ Initial commit created")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to initialize Git repository: {e}")
            return False
    
    def create_github_repo_instructions(self):
        """Provide instructions for creating GitHub repository"""
        print("\n" + "="*60)
        print("GITHUB REPOSITORY CREATION INSTRUCTIONS")
        print("="*60)
        
        instructions = f"""
1. Go to GitHub.com and sign in to your account

2. Click the "+" icon in the top right corner and select "New repository"

3. Fill in the repository details:
   - Repository name: {self.repo_name}
   - Description: DeepCSAT: E-Commerce Customer Satisfaction Score Prediction using Deep Learning
   - Visibility: Public (recommended) or Private
   - Initialize with README: ❌ (we already have one)
   - Add .gitignore: ❌ (we already have one)
   - Choose a license: MIT License (recommended)

4. Click "Create repository"

5. After creating the repository, GitHub will show you the commands to connect your local repository.
   The commands will look like this:

   git remote add origin https://github.com/YOUR_USERNAME/{self.repo_name}.git
   git branch -M main
   git push -u origin main

6. Copy the repository URL and run the following commands in your terminal:
"""
        
        print(instructions)
        
        # Get repository URL from user
        repo_url = input(f"Enter your GitHub repository URL (https://github.com/USERNAME/{self.repo_name}.git): ").strip()
        
        if repo_url:
            self.repo_url = repo_url
            return repo_url
        else:
            print("❌ No repository URL provided")
            return None
    
    def connect_to_github(self, repo_url):
        """Connect local repository to GitHub"""
        print("Connecting to GitHub repository...")
        
        try:
            # Add remote origin
            subprocess.run(['git', 'remote', 'add', 'origin', repo_url], check=True)
            print("✅ Remote origin added")
            
            # Rename branch to main
            subprocess.run(['git', 'branch', '-M', 'main'], check=True)
            print("✅ Branch renamed to main")
            
            # Push to GitHub
            subprocess.run(['git', 'push', '-u', 'origin', 'main'], check=True)
            print("✅ Code pushed to GitHub successfully!")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to connect to GitHub: {e}")
            return False
    
    def create_project_structure(self):
        """Create proper project structure"""
        print("Creating project structure...")
        
        # Create necessary directories
        directories = [
            'data',
            'models',
            'config',
            'results',
            'logs',
            'monitoring',
            'visualizations',
            'docs',
            'tests'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        
        # Create a sample data file
        sample_data_path = 'data/sample_data.csv'
        if not os.path.exists(sample_data_path):
            self.create_sample_data(sample_data_path)
        
        # Create additional documentation
        self.create_additional_docs()
        
        print("✅ Project structure created")
    
    def create_sample_data(self, filepath):
        """Create sample data for testing"""
        import pandas as pd
        import numpy as np
        
        print("Creating sample data...")
        
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
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        df.to_csv(filepath, index=False)
        print(f"✅ Sample data created: {filepath}")
    
    def create_additional_docs(self):
        """Create additional documentation files"""
        
        # Create CONTRIBUTING.md
        contributing_content = """# Contributing to DeepCSAT

Thank you for your interest in contributing to DeepCSAT!

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run setup: `python setup.py`
4. Run tests: `python -m pytest tests/`

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write tests for new features

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.
"""
        
        with open('CONTRIBUTING.md', 'w') as f:
            f.write(contributing_content)
        
        # Create LICENSE
        license_content = """MIT License

Copyright (c) 2023 DeepCSAT Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        
        with open('LICENSE', 'w') as f:
            f.write(license_content)
        
        # Create CHANGELOG.md
        changelog_content = """# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2023-12-01

### Added
- Initial release of DeepCSAT project
- E-commerce customer satisfaction score prediction
- Deep learning ANN model implementation
- Multiple ML algorithms (Random Forest, Gradient Boosting, SVM, etc.)
- Comprehensive NLP processing pipeline
- 15+ data visualizations following UBM analysis
- Statistical hypothesis testing
- Production-ready deployment with Docker
- RESTful API with Flask
- Comprehensive documentation

### Features
- Real-time CSAT score prediction
- Channel optimization insights
- Agent performance analysis
- Sentiment analysis of customer remarks
- Topic modeling and text clustering
- Model persistence and versioning
- Monitoring and health checks
"""
        
        with open('CHANGELOG.md', 'w') as f:
            f.write(changelog_content)
        
        print("✅ Additional documentation created")
    
    def run_setup(self):
        """Run the complete GitHub setup process"""
        print("="*60)
        print("DEEPCSAT GITHUB SETUP")
        print("="*60)
        
        # Step 1: Check git configuration
        if not self.check_git_config():
            if not self.configure_git():
                return False
        
        # Step 2: Create project structure
        self.create_project_structure()
        
        # Step 3: Initialize git repository
        if not self.initialize_git_repo():
            return False
        
        # Step 4: Get GitHub repository URL
        repo_url = self.create_github_repo_instructions()
        if not repo_url:
            return False
        
        # Step 5: Connect to GitHub
        if not self.connect_to_github(repo_url):
            return False
        
        print("\n" + "="*60)
        print("✅ GITHUB SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nYour repository is now available at: {repo_url}")
        print("\nNext steps:")
        print("1. Visit your repository on GitHub")
        print("2. Add a description and topics")
        print("3. Enable GitHub Pages if desired")
        print("4. Set up GitHub Actions for CI/CD")
        print("5. Add collaborators if needed")
        
        return True

def main():
    """Main function"""
    print("DeepCSAT GitHub Repository Setup")
    print("This script will help you create and upload your DeepCSAT project to GitHub")
    
    # Get repository name
    repo_name = input("Enter repository name (default: DeepCSAT): ").strip() or "DeepCSAT"
    
    # Get username
    username = input("Enter your GitHub username: ").strip()
    
    # Create setup instance
    setup = GitHubSetup(repo_name, username)
    
    # Run setup
    if setup.run_setup():
        print("\n🎉 Congratulations! Your DeepCSAT project is now on GitHub!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
