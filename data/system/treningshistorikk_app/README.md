# SkiErgportal

Lokal Streamlit-portal for analyse av Concept2 SkiErg-økter. Appen kan lese lokale Concept2 CSV-eksporter eller hente historikk fra Concept2 API-et.

## Status

Dette er en lokal Python-app, ikke en pakket Windows-app. Se [prosjektets hovedguide](../../README.md) for installasjon, oppstart og personvern.

Portalen inneholder blant annet:

- Økttabell med søk, filtre, klikkbare rader, totaldistanse og kompakt intervalloppsett.
- Leaderboard for standarddistanser, tidsøkter og registrerte intervallprotokoller.
- Øktdetaljer med kommentarer, tagger, kroppsvekt, laktat per drag og terskeltestanalyse.
- Analyseflater for volum, effekt, puls, pace og laktat.

## Kjør lokalt

### Start via launcher

Kjør [Treningshistorikk.cmd](../../Treningshistorikk.cmd) fra rotmappen. Launcheren kan opprette `system/.venv` og installere avhengigheter første gang.

### Start fra terminal

Fra `system/`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r treningshistorikk_app/requirements.txt
streamlit run treningshistorikk_app/app.py
```

Appen åpner normalt på `http://127.0.0.1:8501` og finner `system/Data/` automatisk.

## Datakilder

### Lokale CSV-filer

Legg Concept2 Logbook CSV-eksporter med navnet `concept2-season-*.csv` i `system/Data/`. Velg **CSV (system/Data)** i sidepanelet. Alle sesongfilene leses samtidig; avgrens deretter med portalfiltrene.

CSV-filer kan lastes opp for den aktive nettleserøkten. Opplastede filer lagres ikke automatisk i `system/Data/`.

### Concept2 API

Velg **Concept2 API** i sidepanelet og legg inn en personlig, read-only access token. Tokenet blir ikke forhåndsutfylt eller lagret av portalen.

API-resultater mellomlagres lokalt i `system/Data/api_cache_skierg.json` for raskere senere oppstart.

## Lokal data og personvern

Lokale CSV-filer, API-cache, innstillinger, kommentarer, tagger, laktat og vektdata ligger under `system/Data/`. Ikke commit private data eller token til Git.

Ikke lagre token i kode, Streamlit-konfigurasjon eller filer som deles. Tokenet oppgis manuelt for den aktive portaløkten.

## Avhengigheter

Avhengigheter er definert i [requirements.txt](requirements.txt). De viktigste er Streamlit, pandas, Plotly, NumPy, requests, Pillow og `streamlit-aggrid`.
