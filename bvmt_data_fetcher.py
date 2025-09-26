import requests
import pandas as pd
import json

def fetch_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Si l'API retourne un dictionnaire avec une clé 'markets'
        if 'markets' in data:
            df = pd.DataFrame(data['markets'])
        else:
            df = pd.DataFrame(data)
        
        if not df.empty:
            df.columns = df.columns.str.lower()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(
                        lambda x: str(x).replace(",", ".") if isinstance(x, str) else x
                    )
        return df
    except Exception as e:
        print(f"Erreur lors du chargement de {url}: {e}")
        return pd.DataFrame()

def parse_markets_data(df):
    if df.empty:
        return df
    
    # Si les données sont déjà normalisées (cas des APIs BVMT)
    if 'referentiel' in df.columns:
        # Extraire les informations du référentiel
        df['company_name'] = df['referentiel'].apply(lambda x: x.get('stockName', '') if isinstance(x, dict) else '')
        df['ticker'] = df['referentiel'].apply(lambda x: x.get('ticker', '') if isinstance(x, dict) else '')
        df['arab_name'] = df['referentiel'].apply(lambda x: x.get('arabName', '') if isinstance(x, dict) else '')
        df['val_group'] = df['referentiel'].apply(lambda x: x.get('valGroup', '') if isinstance(x, dict) else '')
    
    return df

URL_HAUSSES = "https://bvmt.com.tn/rest_api/rest/market/hausses"
URL_BAISSES = "https://bvmt.com.tn/rest_api/rest/market/baisses"
URL_VOLUMES = "https://bvmt.com.tn/rest_api/rest/market/volumes"
URL_QTYS = "https://bvmt.com.tn/rest_api/rest/market/qtys"
URL_GROUPS = "https://bvmt.com.tn/rest_api/rest/market/groups/11,12,51,52,99"

def get_all_bvmt_data():
    df_hausses = fetch_data(URL_HAUSSES)
    df_baisses = fetch_data(URL_BAISSES)
    df_volumes = fetch_data(URL_VOLUMES)
    df_qtys = fetch_data(URL_QTYS)
    df_groups = fetch_data(URL_GROUPS)
    
    df_hausses = parse_markets_data(df_hausses)
    df_baisses = parse_markets_data(df_baisses)
    df_volumes = parse_markets_data(df_volumes)
    df_qtys = parse_markets_data(df_qtys)
    df_groups = parse_markets_data(df_groups)
    
    return df_hausses, df_baisses, df_volumes, df_qtys, df_groups

if __name__ == "__main__":
    df_h, df_b, df_v, df_q, df_g = get_all_bvmt_data()
    print("Hausses data shape:", df_h.shape)
    print("Baisses data shape:", df_b.shape)
    print("Volumes data shape:", df_v.shape)
    print("Quantities data shape:", df_q.shape)
    print("Groups data shape:", df_g.shape)
    
    if not df_h.empty:
        print("\nHausses data columns:", df_h.columns.tolist())
        print("\nPremières lignes Hausses:")
        print(df_h[['ticker', 'last', 'change', 'volume']].head())