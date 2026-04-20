import re
import warnings
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# High-risk keywords
RISK_KEYWORDS = [
    'urgent', 'verify', 'login', 'password', 'bank', 
    'suspend', 'account', 'invoice', 'payment', 'winner'
]

EMBEDDING_DIM = 768

# NOTE: BERT support is disabled in this environment because torch/transformers imports
# fail on this Windows setup. This keeps the app/train pipeline running using only
# heuristic features.

def get_bert_embeddings(text_series):
    """
    Returns a fixed-size embedding array for each text entry.
    BERT is disabled by default on this system to avoid the Windows DLL import error.
    """
    return np.zeros((len(text_series), EMBEDDING_DIM), dtype=np.float32)

def clean_text(text):
    if not isinstance(text, str): return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_custom_features(df):
    """Extracts numerical cybersecurity heuristics."""
    features = pd.DataFrame(index=df.index)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    features['url_count'] = df['text'].apply(lambda x: len(re.findall(url_pattern, str(x))))
    features['email_length'] = df['text'].apply(lambda x: len(str(x)))
    features['risk_score'] = df['text'].apply(lambda x: sum(1 for word in RISK_KEYWORDS if word in str(x).lower()))
    features['has_html'] = df['text'].apply(lambda x: 1 if bool(BeautifulSoup(str(x), "html.parser").find()) else 0)
    
    return features

def explain_prediction(text, risk_keywords=None):
    """
    Provides human-readable explanation for the phishing detection result.
    Returns a dictionary with detected risks and warnings.
    """
    if risk_keywords is None:
        risk_keywords = RISK_KEYWORDS
    
    explanation = {
        'risks_found': [],
        'risk_level': 'LOW',
        'confidence_factors': []
    }
    
    text_lower = text.lower()
    
    # Check for URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    if len(urls) > 3:
        explanation['risks_found'].append(f"Multiple links detected ({len(urls)} URLs)")
        explanation['confidence_factors'].append('high_url_count')
    elif len(urls) > 0:
        explanation['risks_found'].append(f"Contains {len(urls)} hyperlink(s)")
        explanation['confidence_factors'].append('url_present')
    
    # Check for risky keywords
    found_keywords = [kw for kw in risk_keywords if kw in text_lower]
    if found_keywords:
        explanation['risks_found'].append(f"Contains urgent language: {', '.join(found_keywords[:3])}")
        explanation['confidence_factors'].append('risky_keywords')
    
    # Check for HTML content
    soup = BeautifulSoup(text, 'html.parser')
    if soup.find():
        explanation['risks_found'].append("HTML/formatted content detected")
        explanation['confidence_factors'].append('html_content')
    
    # Check for suspicious patterns
    if re.search(r'verify\s+(?:account|password|identity)', text_lower):
        explanation['risks_found'].append("Credential verification request detected")
        explanation['confidence_factors'].append('credential_request')
    
    if re.search(r'(?:click|act|confirm|update)\s+(?:now|immediately|urgently)', text_lower):
        explanation['risks_found'].append("Urgent action request detected")
        explanation['confidence_factors'].append('urgent_action')
    
    # Determine risk level
    num_risks = len(explanation['risks_found'])
    if num_risks >= 3:
        explanation['risk_level'] = 'CRITICAL'
    elif num_risks >= 2:
        explanation['risk_level'] = 'HIGH'
    elif num_risks >= 1:
        explanation['risk_level'] = 'MEDIUM'
    
    return explanation