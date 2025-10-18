# DeepCSAT: Natural Language Processing for Customer Remarks
# Comprehensive NLP pipeline for text analysis and feature extraction

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DeepCSATNLPProcessor:
    """
    Comprehensive NLP processor for DeepCSAT project
    Handles text preprocessing, feature extraction, and sentiment analysis
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.processed_text = None
        self.tfidf_matrix = None
        self.sentiment_scores = None
        self.topic_model = None
        self.vectorizer = None
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Get stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def preprocess_text(self, text_column='Customer Remarks'):
        """Comprehensive text preprocessing pipeline"""
        print("Starting NLP preprocessing...")
        
        # 1. Expand contractions
        self._expand_contractions(text_column)
        
        # 2. Convert to lowercase
        self._to_lowercase(text_column)
        
        # 3. Remove special characters and numbers
        self._remove_special_characters(text_column)
        
        # 4. Remove URLs
        self._remove_urls(text_column)
        
        # 5. Remove stopwords
        self._remove_stopwords(text_column)
        
        # 6. Tokenization
        self._tokenize_text(text_column)
        
        # 7. Lemmatization
        self._lemmatize_text(text_column)
        
        # 8. Remove extra whitespaces
        self._remove_whitespaces(text_column)
        
        print("NLP preprocessing completed!")
    
    def _expand_contractions(self, text_column):
        """Expand common contractions in text"""
        print("Expanding contractions...")
        
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'s": " is", "'d": " would",
            "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
        }
        
        def expand_contractions_text(text):
            if pd.isna(text):
                return text
            text = str(text)
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            return text
        
        self.df[f'{text_column}_expanded'] = self.df[text_column].apply(expand_contractions_text)
    
    def _to_lowercase(self, text_column):
        """Convert text to lowercase"""
        print("Converting to lowercase...")
        self.df[f'{text_column}_lower'] = self.df[f'{text_column}_expanded'].str.lower()
    
    def _remove_special_characters(self, text_column):
        """Remove special characters and numbers"""
        print("Removing special characters and numbers...")
        
        def clean_text(text):
            if pd.isna(text):
                return text
            # Keep only letters and spaces
            text = re.sub(r'[^a-zA-Z\s]', '', str(text))
            return text
        
        self.df[f'{text_column}_cleaned'] = self.df[f'{text_column}_lower'].apply(clean_text)
    
    def _remove_urls(self, text_column):
        """Remove URLs from text"""
        print("Removing URLs...")
        
        def remove_urls_text(text):
            if pd.isna(text):
                return text
            text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
            return text
        
        self.df[f'{text_column}_no_urls'] = self.df[f'{text_column}_cleaned'].apply(remove_urls_text)
    
    def _remove_stopwords(self, text_column):
        """Remove stopwords from text"""
        print("Removing stopwords...")
        
        def remove_stopwords_text(text):
            if pd.isna(text):
                return text
            words = word_tokenize(str(text))
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
        
        self.df[f'{text_column}_no_stopwords'] = self.df[f'{text_column}_no_urls'].apply(remove_stopwords_text)
    
    def _tokenize_text(self, text_column):
        """Tokenize text into words"""
        print("Tokenizing text...")
        
        def tokenize_text(text):
            if pd.isna(text):
                return []
            return word_tokenize(str(text))
        
        self.df[f'{text_column}_tokens'] = self.df[f'{text_column}_no_stopwords'].apply(tokenize_text)
    
    def _lemmatize_text(self, text_column):
        """Lemmatize words in text"""
        print("Lemmatizing text...")
        
        def lemmatize_text(text):
            if pd.isna(text):
                return text
            words = word_tokenize(str(text))
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        
        self.df[f'{text_column}_lemmatized'] = self.df[f'{text_column}_no_stopwords'].apply(lemmatize_text)
    
    def _remove_whitespaces(self, text_column):
        """Remove extra whitespaces"""
        print("Removing extra whitespaces...")
        
        def clean_whitespaces(text):
            if pd.isna(text):
                return text
            return ' '.join(str(text).split())
        
        self.df[f'{text_column}_final'] = self.df[f'{text_column}_lemmatized'].apply(clean_whitespaces)
        self.processed_text = self.df[f'{text_column}_final']
    
    def extract_text_features(self):
        """Extract various text features"""
        print("Extracting text features...")
        
        # 1. Text length features
        self.df['text_length'] = self.processed_text.str.len()
        self.df['word_count'] = self.processed_text.str.split().str.len()
        self.df['sentence_count'] = self.processed_text.str.count(r'[.!?]+')
        
        # 2. Character features
        self.df['avg_word_length'] = self.df['text_length'] / (self.df['word_count'] + 1)
        self.df['uppercase_ratio'] = self.processed_text.str.count(r'[A-Z]') / (self.df['text_length'] + 1)
        self.df['digit_ratio'] = self.processed_text.str.count(r'[0-9]') / (self.df['text_length'] + 1)
        
        # 3. Punctuation features
        self.df['exclamation_count'] = self.processed_text.str.count('!')
        self.df['question_count'] = self.processed_text.str.count('?')
        self.df['punctuation_ratio'] = self.processed_text.str.count(r'[^\w\s]') / (self.df['text_length'] + 1)
        
        print("Text features extracted!")
    
    def analyze_sentiment(self):
        """Perform sentiment analysis"""
        print("Analyzing sentiment...")
        
        def get_sentiment_scores(text):
            if pd.isna(text) or text == '':
                return {'polarity': 0, 'subjectivity': 0}
            
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        
        # Apply sentiment analysis
        sentiment_scores = self.processed_text.apply(get_sentiment_scores)
        
        # Extract polarity and subjectivity
        self.df['sentiment_polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        self.df['sentiment_subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        
        # Create sentiment categories
        self.df['sentiment_category'] = pd.cut(
            self.df['sentiment_polarity'], 
            bins=[-1, -0.1, 0.1, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        self.sentiment_scores = {
            'polarity': self.df['sentiment_polarity'],
            'subjectivity': self.df['sentiment_subjectivity'],
            'category': self.df['sentiment_category']
        }
        
        print("Sentiment analysis completed!")
    
    def perform_topic_modeling(self, n_topics=5):
        """Perform topic modeling using LDA"""
        print(f"Performing topic modeling with {n_topics} topics...")
        
        # Prepare text data
        text_data = self.processed_text.fillna('').tolist()
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(text_data)
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(self.tfidf_matrix)
        
        # Get topic assignments
        topic_assignments = lda.transform(self.tfidf_matrix)
        self.df['dominant_topic'] = np.argmax(topic_assignments, axis=1)
        self.df['topic_confidence'] = np.max(topic_assignments, axis=1)
        
        # Store topic model
        self.topic_model = lda
        
        print("Topic modeling completed!")
        
        return lda, topic_assignments
    
    def perform_text_clustering(self, n_clusters=5):
        """Perform text clustering using K-means"""
        print(f"Performing text clustering with {n_clusters} clusters...")
        
        if self.tfidf_matrix is None:
            self.perform_topic_modeling()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.tfidf_matrix)
        
        self.df['text_cluster'] = cluster_labels
        
        print("Text clustering completed!")
        
        return kmeans, cluster_labels
    
    def create_visualizations(self):
        """Create comprehensive NLP visualizations"""
        print("Creating NLP visualizations...")
        
        # 1. Sentiment Distribution
        self._plot_sentiment_distribution()
        
        # 2. Text Length Analysis
        self._plot_text_length_analysis()
        
        # 3. Word Cloud
        self._create_word_cloud()
        
        # 4. Topic Visualization
        self._plot_topic_distribution()
        
        # 5. Sentiment vs CSAT
        self._plot_sentiment_vs_csat()
        
        print("NLP visualizations completed!")
    
    def _plot_sentiment_distribution(self):
        """Plot sentiment distribution"""
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Sentiment Category Distribution
        plt.subplot(1, 3, 1)
        sentiment_counts = self.df['sentiment_category'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Sentiment Category Distribution', fontweight='bold')
        
        # Subplot 2: Polarity Distribution
        plt.subplot(1, 3, 2)
        plt.hist(self.df['sentiment_polarity'], bins=30, alpha=0.7, color='skyblue')
        plt.title('Sentiment Polarity Distribution', fontweight='bold')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        
        # Subplot 3: Subjectivity Distribution
        plt.subplot(1, 3, 3)
        plt.hist(self.df['sentiment_subjectivity'], bins=30, alpha=0.7, color='lightcoral')
        plt.title('Sentiment Subjectivity Distribution', fontweight='bold')
        plt.xlabel('Subjectivity Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_text_length_analysis(self):
        """Plot text length analysis"""
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Text Length Distribution
        plt.subplot(1, 3, 1)
        plt.hist(self.df['text_length'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('Text Length Distribution', fontweight='bold')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        
        # Subplot 2: Word Count Distribution
        plt.subplot(1, 3, 2)
        plt.hist(self.df['word_count'], bins=50, alpha=0.7, color='orange')
        plt.title('Word Count Distribution', fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        # Subplot 3: Average Word Length
        plt.subplot(1, 3, 3)
        plt.hist(self.df['avg_word_length'], bins=30, alpha=0.7, color='purple')
        plt.title('Average Word Length Distribution', fontweight='bold')
        plt.xlabel('Average Word Length')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def _create_word_cloud(self):
        """Create word cloud visualization"""
        try:
            from wordcloud import WordCloud
            
            # Combine all text
            all_text = ' '.join(self.processed_text.fillna('').astype(str))
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(all_text)
            
            # Plot word cloud
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Customer Remarks', fontweight='bold', fontsize=16)
            plt.show()
            
        except ImportError:
            print("WordCloud library not available. Install with: pip install wordcloud")
    
    def _plot_topic_distribution(self):
        """Plot topic distribution"""
        if 'dominant_topic' in self.df.columns:
            plt.figure(figsize=(12, 6))
            
            # Topic distribution
            topic_counts = self.df['dominant_topic'].value_counts().sort_index()
            bars = plt.bar(topic_counts.index, topic_counts.values, color='lightblue')
            plt.title('Topic Distribution', fontweight='bold')
            plt.xlabel('Topic ID')
            plt.ylabel('Number of Documents')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
    
    def _plot_sentiment_vs_csat(self):
        """Plot sentiment vs CSAT score relationship"""
        if 'CSAT Score' in self.df.columns:
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Sentiment Category vs CSAT
            plt.subplot(1, 3, 1)
            sentiment_csat = self.df.groupby('sentiment_category')['CSAT Score'].mean()
            bars = plt.bar(sentiment_csat.index, sentiment_csat.values, color=['red', 'yellow', 'green'])
            plt.title('Average CSAT Score by Sentiment', fontweight='bold')
            plt.xlabel('Sentiment Category')
            plt.ylabel('Average CSAT Score')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            # Subplot 2: Polarity vs CSAT Scatter
            plt.subplot(1, 3, 2)
            plt.scatter(self.df['sentiment_polarity'], self.df['CSAT Score'], alpha=0.6, color='blue')
            plt.title('Sentiment Polarity vs CSAT Score', fontweight='bold')
            plt.xlabel('Sentiment Polarity')
            plt.ylabel('CSAT Score')
            
            # Add trend line
            z = np.polyfit(self.df['sentiment_polarity'].dropna(), self.df['CSAT Score'], 1)
            p = np.poly1d(z)
            plt.plot(self.df['sentiment_polarity'], p(self.df['sentiment_polarity']), "r--", alpha=0.8)
            
            # Subplot 3: Subjectivity vs CSAT Scatter
            plt.subplot(1, 3, 3)
            plt.scatter(self.df['sentiment_subjectivity'], self.df['CSAT Score'], alpha=0.6, color='green')
            plt.title('Sentiment Subjectivity vs CSAT Score', fontweight='bold')
            plt.xlabel('Sentiment Subjectivity')
            plt.ylabel('CSAT Score')
            
            # Add trend line
            z = np.polyfit(self.df['sentiment_subjectivity'].dropna(), self.df['CSAT Score'], 1)
            p = np.poly1d(z)
            plt.plot(self.df['sentiment_subjectivity'], p(self.df['sentiment_subjectivity']), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.show()
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print("Model does not support feature importance")
            return None
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_processed_data(self, filepath='processed_data.csv'):
        """Save processed data with NLP features"""
        self.df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    
    def get_summary_statistics(self):
        """Get summary statistics of NLP features"""
        print("NLP Feature Summary Statistics:")
        print("="*50)
        
        # Text length statistics
        print(f"Average text length: {self.df['text_length'].mean():.2f} characters")
        print(f"Average word count: {self.df['word_count'].mean():.2f} words")
        print(f"Average sentence count: {self.df['sentence_count'].mean():.2f} sentences")
        
        # Sentiment statistics
        print(f"Average sentiment polarity: {self.df['sentiment_polarity'].mean():.3f}")
        print(f"Average sentiment subjectivity: {self.df['sentiment_subjectivity'].mean():.3f}")
        
        # Sentiment distribution
        print("\nSentiment Category Distribution:")
        print(self.df['sentiment_category'].value_counts())
        
        # Topic distribution
        if 'dominant_topic' in self.df.columns:
            print(f"\nTopic Distribution:")
            print(self.df['dominant_topic'].value_counts().sort_index())

# Example usage
if __name__ == "__main__":
    print("DeepCSAT NLP Processor Ready!")
    print("Usage:")
    print("1. Load your dataset: df = pd.read_csv('your_dataset.csv')")
    print("2. Initialize processor: nlp = DeepCSATNLPProcessor(df)")
    print("3. Preprocess text: nlp.preprocess_text('Customer Remarks')")
    print("4. Extract features: nlp.extract_text_features()")
    print("5. Analyze sentiment: nlp.analyze_sentiment()")
    print("6. Create visualizations: nlp.create_visualizations()")
    print("7. Get summary: nlp.get_summary_statistics()")
