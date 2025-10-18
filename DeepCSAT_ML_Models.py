# DeepCSAT: Machine Learning and Deep Learning Models
# Comprehensive ML pipeline for CSAT score prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2
import warnings
warnings.filterwarnings('ignore')

class DeepCSATMLPipeline:
    """
    Comprehensive ML pipeline for DeepCSAT project
    Includes data preprocessing, feature engineering, and multiple ML models
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def data_preprocessing(self):
        """Comprehensive data preprocessing pipeline"""
        print("Starting Data Preprocessing...")
        
        # 1. Handle missing values
        self._handle_missing_values()
        
        # 2. Feature engineering
        self._feature_engineering()
        
        # 3. Categorical encoding
        self._categorical_encoding()
        
        # 4. Feature selection
        self._feature_selection()
        
        # 5. Data splitting
        self._data_splitting()
        
        print("Data Preprocessing Completed!")
        
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Fill missing values in Customer Remarks
        if 'Customer Remarks' in self.df.columns:
            self.df['Customer Remarks'] = self.df['Customer Remarks'].fillna('No remarks provided')
        
        # Fill missing values in numerical columns with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill missing values in categorical columns with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print(f"Missing values handled. Remaining missing values: {self.df.isnull().sum().sum()}")
    
    def _feature_engineering(self):
        """Create new features for better model performance"""
        print("Creating engineered features...")
        
        # 1. Time-based features
        if 'order_date_time' in self.df.columns:
            self.df['order_date_time'] = pd.to_datetime(self.df['order_date_time'], errors='coerce')
            self.df['order_hour'] = self.df['order_date_time'].dt.hour
            self.df['order_day_of_week'] = self.df['order_date_time'].dt.day_name()
            self.df['order_month'] = self.df['order_date_time'].dt.month
        
        # 2. Response time calculation
        if 'Issue_reported at' in self.df.columns and 'issue_responded' in self.df.columns:
            self.df['Issue_reported at'] = pd.to_datetime(self.df['Issue_reported at'], errors='coerce')
            self.df['issue_responded'] = pd.to_datetime(self.df['issue_responded'], errors='coerce')
            self.df['response_time_hours'] = (
                self.df['issue_responded'] - self.df['Issue_reported at']
            ).dt.total_seconds() / 3600
        
        # 3. Price categories
        if 'Item_price' in self.df.columns:
            self.df['price_category'] = pd.cut(
                self.df['Item_price'], 
                bins=[0, 100, 500, 1000, float('inf')], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # 4. Handling time categories
        if 'connected_handling_time' in self.df.columns:
            self.df['handling_time_category'] = pd.cut(
                self.df['connected_handling_time'], 
                bins=[0, 5, 15, 30, float('inf')], 
                labels=['Quick', 'Normal', 'Slow', 'Very Slow']
            )
        
        # 5. Customer remarks length
        if 'Customer Remarks' in self.df.columns:
            self.df['remarks_length'] = self.df['Customer Remarks'].str.len()
            self.df['remarks_word_count'] = self.df['Customer Remarks'].str.split().str.len()
        
        print("Feature engineering completed!")
    
    def _categorical_encoding(self):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'CSAT Score':  # Skip target variable
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        print("Categorical encoding completed!")
    
    def _feature_selection(self):
        """Select most important features"""
        print("Performing feature selection...")
        
        # Prepare features and target
        feature_cols = [col for col in self.df.columns if col != 'CSAT Score']
        X = self.df[feature_cols]
        y = self.df['CSAT Score']
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=15)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected features: {selected_features}")
        
        # Update dataframe with selected features
        self.df = self.df[selected_features + ['CSAT Score']]
        
        print("Feature selection completed!")
    
    def _data_splitting(self):
        """Split data into train and test sets"""
        print("Splitting data into train and test sets...")
        
        # Prepare features and target
        feature_cols = [col for col in self.df.columns if col != 'CSAT Score']
        X = self.df[feature_cols]
        y = self.df['CSAT Score']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple ML models"""
        print("Training Multiple ML Models...")
        
        # Model 1: Random Forest Regressor
        self._train_random_forest()
        
        # Model 2: Gradient Boosting Regressor
        self._train_gradient_boosting()
        
        # Model 3: Support Vector Regressor
        self._train_svr()
        
        # Model 4: Linear Regression
        self._train_linear_regression()
        
        # Model 5: Deep Learning ANN
        self._train_deep_learning_ann()
        
        print("All models trained successfully!")
    
    def _train_random_forest(self):
        """Train Random Forest Regressor with hyperparameter tuning"""
        print("Training Random Forest Regressor...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.models['Random Forest'] = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = self.models['Random Forest'].predict(self.X_train)
        y_pred_test = self.models['Random Forest'].predict(self.X_test)
        
        # Calculate metrics
        self.results['Random Forest'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'best_params': grid_search.best_params_
        }
        
        print(f"Random Forest - Test R²: {self.results['Random Forest']['test_r2']:.4f}")
    
    def _train_gradient_boosting(self):
        """Train Gradient Boosting Regressor with hyperparameter tuning"""
        print("Training Gradient Boosting Regressor...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.models['Gradient Boosting'] = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = self.models['Gradient Boosting'].predict(self.X_train)
        y_pred_test = self.models['Gradient Boosting'].predict(self.X_test)
        
        # Calculate metrics
        self.results['Gradient Boosting'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'best_params': grid_search.best_params_
        }
        
        print(f"Gradient Boosting - Test R²: {self.results['Gradient Boosting']['test_r2']:.4f}")
    
    def _train_svr(self):
        """Train Support Vector Regressor with hyperparameter tuning"""
        print("Training Support Vector Regressor...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Grid search
        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Best model
        self.models['SVR'] = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = self.models['SVR'].predict(self.X_train_scaled)
        y_pred_test = self.models['SVR'].predict(self.X_test_scaled)
        
        # Calculate metrics
        self.results['SVR'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'best_params': grid_search.best_params_
        }
        
        print(f"SVR - Test R²: {self.results['SVR']['test_r2']:.4f}")
    
    def _train_linear_regression(self):
        """Train Linear Regression models"""
        print("Training Linear Regression models...")
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_train_scaled, self.y_train)
        self.models['Ridge Regression'] = ridge
        
        # Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(self.X_train_scaled, self.y_train)
        self.models['Lasso Regression'] = lasso
        
        # Calculate metrics for all linear models
        for name, model in [('Linear Regression', lr), ('Ridge Regression', ridge), ('Lasso Regression', lasso)]:
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            self.results[name] = {
                'train_r2': r2_score(self.y_train, y_pred_train),
                'test_r2': r2_score(self.y_test, y_pred_test),
                'train_mae': mean_absolute_error(self.y_train, y_pred_train),
                'test_mae': mean_absolute_error(self.y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            }
            
            print(f"{name} - Test R²: {self.results[name]['test_r2']:.4f}")
    
    def _train_deep_learning_ann(self):
        """Train Deep Learning ANN model"""
        print("Training Deep Learning ANN...")
        
        # Build the model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='linear')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        # Train the model
        history = model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_test_scaled, self.y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.models['Deep Learning ANN'] = model
        
        # Predictions
        y_pred_train = model.predict(self.X_train_scaled).flatten()
        y_pred_test = model.predict(self.X_test_scaled).flatten()
        
        # Calculate metrics
        self.results['Deep Learning ANN'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'history': history
        }
        
        print(f"Deep Learning ANN - Test R²: {self.results['Deep Learning ANN']['test_r2']:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("Evaluating and comparing all models...")
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['train_r2', 'test_r2', 'train_mae', 'test_mae', 'train_rmse', 'test_rmse']]
        
        print("\nModel Performance Comparison:")
        print("="*80)
        print(results_df.round(4))
        
        # Find best model
        best_model = results_df['test_r2'].idxmax()
        print(f"\nBest Model: {best_model}")
        print(f"Best Test R² Score: {results_df.loc[best_model, 'test_r2']:.4f}")
        
        # Plot model comparison
        self._plot_model_comparison(results_df)
        
        return results_df
    
    def _plot_model_comparison(self, results_df):
        """Plot model comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        axes[0, 0].bar(results_df.index, results_df['test_r2'], color='skyblue')
        axes[0, 0].set_title('Model R² Score Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(results_df.index, results_df['test_mae'], color='lightcoral')
        axes[0, 1].set_title('Model MAE Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[1, 0].bar(results_df.index, results_df['test_rmse'], color='lightgreen')
        axes[1, 0].set_title('Model RMSE Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training vs Test R²
        axes[1, 1].scatter(results_df['train_r2'], results_df['test_r2'], s=100, alpha=0.7)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[1, 1].set_xlabel('Training R²')
        axes[1, 1].set_ylabel('Test R²')
        axes[1, 1].set_title('Training vs Test R²', fontweight='bold')
        
        # Add model names as annotations
        for i, model in enumerate(results_df.index):
            axes[1, 1].annotate(model, (results_df['train_r2'].iloc[i], results_df['test_r2'].iloc[i]))
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath='models/'):
        """Save trained models"""
        import os
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        # Save traditional ML models
        for name, model in self.models.items():
            if name != 'Deep Learning ANN':
                joblib.dump(model, f"{filepath}{name.replace(' ', '_').lower()}.joblib")
        
        # Save deep learning model
        if 'Deep Learning ANN' in self.models:
            self.models['Deep Learning ANN'].save(f"{filepath}deep_learning_ann.h5")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, f"{filepath}scaler.joblib")
        joblib.dump(self.label_encoders, f"{filepath}label_encoders.joblib")
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models"""
        import joblib
        
        # Load traditional ML models
        for name in ['Random Forest', 'Gradient Boosting', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            try:
                self.models[name] = joblib.load(f"{filepath}{name.replace(' ', '_').lower()}.joblib")
            except FileNotFoundError:
                print(f"Model {name} not found")
        
        # Load deep learning model
        try:
            self.models['Deep Learning ANN'] = tf.keras.models.load_model(f"{filepath}deep_learning_ann.h5")
        except FileNotFoundError:
            print("Deep Learning model not found")
        
        # Load scaler and encoders
        try:
            self.scaler = joblib.load(f"{filepath}scaler.joblib")
            self.label_encoders = joblib.load(f"{filepath}label_encoders.joblib")
        except FileNotFoundError:
            print("Scaler or encoders not found")
    
    def predict(self, X, model_name='Deep Learning ANN'):
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'Deep Learning ANN':
            X_scaled = self.scaler.transform(X)
            predictions = model.predict(X_scaled).flatten()
        else:
            if model_name in ['SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_scaled = self.scaler.transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
        
        return predictions

# Example usage
if __name__ == "__main__":
    print("DeepCSAT ML Pipeline Ready!")
    print("Usage:")
    print("1. Load your dataset: df = pd.read_csv('your_dataset.csv')")
    print("2. Initialize pipeline: pipeline = DeepCSATMLPipeline(df)")
    print("3. Preprocess data: pipeline.data_preprocessing()")
    print("4. Train models: pipeline.train_models()")
    print("5. Evaluate models: pipeline.evaluate_models()")
    print("6. Save models: pipeline.save_models()")
