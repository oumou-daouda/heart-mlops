import pandas as pd

# Noms des colonnes du dataset Cleveland
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def load_data(filepath):
    """Charge le dataset et retourne un DataFrame brut."""
    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    return df

def clean_data(df):
    """Remplace les valeurs manquantes et convertit en float."""
    df = df.replace('?', pd.NA).dropna()
    df = df.astype(float)
    return df

def prepare_target(df):
    """Binarise la variable cible : 0 = pas de maladie, 1 = maladie."""
    df['target'] = (df['target'] > 0).astype(int)
    return df

def preprocess(filepath):
    """Pipeline complet : charge, nettoie et prépare les données."""
    df = load_data(filepath)
    df = clean_data(df)
    df = prepare_target(df)
    return df

'''if __name__ == "__main__":
    data = preprocess('data/heart.csv')
    print(f"Dataset prêt : {data.shape}")
    print(data.head())
    '''
if __name__ == "__main__":
    data = preprocess('data/heart.csv')
    # Sauvegarder le dataset nettoyé pour le pipeline DVC
    data.to_csv('data/heart_clean.csv', index=False)
    print(f"Dataset prêt : {data.shape}")
    print(data.head())