import sys
sys.path.append('.')

import pytest
import pandas as pd
from src.preprocessing import load_data, clean_data, prepare_target, preprocess

# --- Tests sur load_data ---

def test_load_data_shape():
    """Le dataset doit avoir 303 lignes et 14 colonnes avant nettoyage."""
    df = load_data('data/heart.csv')
    assert df.shape == (303, 14)

def test_load_data_columns():
    """Les colonnes doivent correspondre aux variables cliniques attendues."""
    df = load_data('data/heart.csv')
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol',
                     'fbs', 'restecg', 'thalach', 'exang',
                     'oldpeak', 'slope', 'ca', 'thal', 'target']
    assert list(df.columns) == expected_cols

# --- Tests sur clean_data ---

def test_clean_data_no_missing():
    """Après nettoyage, il ne doit plus y avoir de valeurs manquantes."""
    df = load_data('data/heart.csv')
    df = clean_data(df)
    assert df.isnull().sum().sum() == 0

def test_clean_data_numeric():
    """Toutes les colonnes doivent être numériques après nettoyage."""
    df = load_data('data/heart.csv')
    df = clean_data(df)
    assert all(df.dtypes == float)

def test_clean_data_rows():
    """Après nettoyage, on doit avoir 297 lignes (6 supprimées)."""
    df = load_data('data/heart.csv')
    df = clean_data(df)
    assert df.shape[0] == 297

# --- Tests sur prepare_target ---

def test_target_binary():
    """La variable cible doit être binaire — uniquement 0 et 1."""
    df = preprocess('data/heart.csv')
    assert set(df['target'].unique()).issubset({0, 1})

def test_target_distribution():
    """Le dataset doit être approximativement équilibré."""
    df = preprocess('data/heart.csv')
    counts = df['target'].value_counts()
    assert counts[0] == 160
    assert counts[1] == 137

# --- Test pipeline complet ---

def test_preprocess_pipeline():
    """Le pipeline complet doit retourner un DataFrame propre et prêt."""
    df = preprocess('data/heart.csv')
    assert df.shape == (297, 14)
    assert df.isnull().sum().sum() == 0
    assert set(df['target'].unique()).issubset({0, 1})