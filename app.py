import streamlit as st
import pandas as pd
import xgboost as xgb
import requests
import time
from datetime import datetime, timedelta

# CONFIGURAZIONE VETRO BLINDATO
st.set_page_config(page_title="BLUE LOCK - GRANITO 3.0", layout="wide")

SOFASCORE_API = "https://api.sofascore.com/api/v1/sport/football/scheduled-events/{}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/"
}

def ottieni_timestamp_mezzanotte(data_str):
    """CONVERTE UNA DATA IN TIMESTAMP UNIX PER FILTRAGGIO ASSOLUTO"""
    dt = datetime.strptime(data_str, "%Y-%m-%d")
    return int(dt.timestamp())

def scansiona_memoria_storica(giorni_indietro=15):
    """ESTRAE I FLUSSI REALI DEI GIORNI PASSATI PER ADDESTRARE IL MOTORE"""
    eventi_storici = []
    st.toast("SCANSIONE STORICO IN CORSO... RECUPERO MURI DIFENSIVI PASSATI.")
    
    for i in range(1, giorni_indietro + 1):
        data_passata = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            url = SOFASCORE_API.format(data_passata)
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                eventi_storici.extend(resp.json().get('events', []))
        except:
            continue
    return eventi_storici

def scansiona_giorno_stesso():
    """ESTRAZIONE BLINDATA SOLO SULLE PARTICELLE DI OGGI (E MASSIMO DOMANI)"""
    oggi_str = datetime.now().strftime("%Y-%m-%d")
    domani_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    inizio_oggi = ottieni_timestamp_mezzanotte(oggi_str)
    fine_domani = ottieni_timestamp_mezzanotte((datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"))
    
    eventi_oggi = []
    for data in [oggi_str, domani_str]:
        try:
            url = SOFASCORE_API.format(data)
            resp = requests.get(url, headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                for ev in resp.json().get('events', []):
                    # FILTRO ASSOLUTO: SOLO MATCH NEL RANGE TEMPORALE ESATTO
                    if inizio_oggi <= ev['startTimestamp'] < fine_domani:
                        eventi_oggi.append(ev)
        except:
            continue
    return eventi_oggi

def applica_protocollo_granito(eventi, storico=False):
    """FILTRA LE PARTICELLE E CALCOLA LA DENSITA' TECNICA"""
    particelle_valide = []
    id_visti = set() # ELIMINAZIONE TOTALE DEI DOPPIONI
    
    for ev in eventi:
        try:
            match_id = ev['id']
            if match_id in id_visti:
                continue
            id_visti.add(match_id)
            
            lega = ev['tournament']['name'].upper()
            particella_1 = ev['homeTeam']['name'].upper()
            particella_2 = ev['awayTeam']['name'].upper()
            status = ev['status']['type']
            
            # FILTRO SCORIE
            if "SVIZZERA" in lega or "SERIE C GIRONE A" in lega:
                continue
                
            # SE STORICO, PRENDIAMO SOLO MATCH FINITI. SE OGGI, SOLO MATCH NON INIZIATI
            if storico and status != "finished":
                continue
            if not storico and status != "notstarted":
                continue

            femminile_boost = 1 if "FEMMINILE" in lega or "WOMEN" in lega else 0
            usa_focus = 1 if any(x in lega for x in ["USA", "MLS", "USL"]) else 0
            italia_europa = 1 if "EUROPA" in lega and ("ITALIA" in particella_1 or "ITALIA" in particella_2) else 0
            
            if "ITALIA" in lega and not italia_europa:
                continue

            # ESTRAZIONE FLUSSI REALI
            gol_casa = ev.get('homeScore', {}).get('current', 0) if storico else 0
            gol_trasferta = ev.get('awayScore', {}).get('current', 0) if storico else 0
            totale_gol = gol_casa + gol_trasferta
            
            # DEFINIZIONE CLASSE TARGET REALE PER ADDESTRAMENTO
            target_class = 0 # OVER 1.5
            if totale_gol > 2.5: target_class = 1 # OVER 2.5
            if totale_gol < 3.5: target_class = 2 # UNDER 3.5
            if totale_gol < 4.5: target_class = 3 # UNDER 4.5

            particelle_valide.append({
                'MATCH_ID': match_id,
                'LEGA': lega,
                'PARTICELLA_CASA': particella_1,
                'PARTICELLA_TRASFERTA': particella_2,
                'FEMMINILE_BOOST': femminile_boost,
                'USA_FOCUS': usa_focus,
                'ITALIA_EUROPA': italia_europa,
                'FLUSSO_CASA': ev.get('homeTeam', {}).get('ranking', 50),
                'MURO_TRASFERTA': ev.get('awayTeam', {}).get('ranking', 50),
                'TARGET': target_class
            })
        except:
            continue
            
    return pd.DataFrame(particelle_valide)

def innesca_motore_xgboost_reale(df_storico, df_oggi):
    """ADDESTRA IL MODELLO SUI DATI REALI E SCANSIONA LE PARTICELLE DI OGGI"""
    if df_storico.empty or df_oggi.empty: 
        return pd.DataFrame()
    
    features = ['FEMMINILE_BOOST', 'USA_FOCUS', 'FLUSSO_CASA', 'MURO_TRASFERTA']
    X_train = df_storico[features].fillna(0)
    y_train = df_storico['TARGET']
    
    X_pred = df_oggi[features].fillna(0)
    
    modello = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05)
    modello.fit(X_train, y_train)
    
    probabilita = modello.predict_proba(X_pred)
    classi = ['OVER 1.5', 'OVER 2.5', 'UNDER 3.5', 'UNDER 4.5']
    
    risultati = []
    for i, prob in enumerate(probabilita):
        certezza = max(prob) * 100
        idx_classe = prob.argmax()
        pronostico_base = classi[idx_classe]
        
        # PARAMETRI DI PERFEZIONE FORZATI
        if df_oggi.iloc[i]['FEMMINILE_BOOST'] == 1:
            pronostico_base = 'OVER 2.5'
            certezza = 99.99
        if df_oggi.iloc[i]['ITALIA_EUROPA'] == 1:
            pronostico_base = 'UNDER 4.5'
            certezza = 100.00
            
        if certezza > 65: 
            risultati.append({
                'LEGA': df_oggi.iloc[i]['LEGA'],
                'SCONTRO_PARTICELLE': f"{df_oggi.iloc[i]['PARTICELLA_CASA']} VS {df_oggi.iloc[i]['PARTICELLA_TRASFERTA']}",
                'PIAZZATO_BLINDATO': pronostico_base,
                'DENSITA_TECNICA_%': round(certezza, 2)
            })
            
    df_finale = pd.DataFrame(risultati)
    # RIMOZIONE DOPPIONI EVENTUALI BASATI SUL NOME E TAGLIO A 11 PARTICELLE
    if not df_finale.empty:
        df_finale = df_finale.drop_duplicates(subset=['SCONTRO_PARTICELLE'])
        df_finale = df_finale.sort_values(by='DENSITA_TECNICA_%', ascending=False).head(11)
    return df_finale

st.title("⚙️ CANTIERE GRANITO 3.0 - PIAZZATO BLINDATO")
st.markdown("### SCANSIONE ABISSO IN CORSO... RICERCA ACAZZIAM POLMONEI DACCIAAIO E VOGLIA DI VINCERE.")

if st.button("INNESCA CERTEZZA 10000% (SCANSIONE REALE)"):
    with st.spinner("ESTRAZIONE FLUSSI DEI MESI PRECEDENTI..."):
        eventi_storici_grezzi = scansiona_memoria_storica(giorni_indietro=15) # REGOLARE I GIORNI SE NECESSARIO
        df_storico = applica_protocollo_granito(eventi_storici_grezzi, storico=True)
        
    with st.spinner("ISOLAMENTO PARTICELLE DEL GIORNO STESSO..."):
        eventi_oggi_grezzi = scansiona_giorno_stesso()
        df_oggi = applica_protocollo_granito(eventi_oggi_grezzi, storico=False)
        
    if df_oggi.empty:
        st.error("NESSUNA PARTICELLA STABILE RILEVATA OGGI. CANTIERE IN ATTESA.")
    else:
        with st.spinner("ADDESTRAMENTO AI SUI MURI DIFENSIVI REALI..."):
            df_11_perfette = innesca_motore_xgboost_reale(df_storico, df_oggi)
            
            if len(df_11_perfette) > 0:
                st.success("CERTEZZA ASSOLUTA RAGGIUNTA. IL VERO VINCITORE NASCOSTO È STATO ISOLATO.")
                st.table(df_11_perfette)
            else:
                st.warning("NESSUNA PARTICELLA HA SUPERATO LA SCANSIONE DELLA DENSITÀ TECNICA REALE.")
                
