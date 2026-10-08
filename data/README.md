# SkiErgportal

Lokal Streamlit-app for analyse av SkiErg-okter fra Concept2 Logbook.

## Kom i gang

1. Installer [Python 3.11 eller nyere](https://www.python.org/downloads/windows/). Hvis Python mangler, apner launcheren automatisk nedlastingssiden.
2. Last ned prosjektet eller klon det fra GitHub.
3. Dobbeltklikk `Trykk her for å starte skiergportal.cmd`.
4. Vent mens appen oppretter et lokalt virtuelt milj&#248; og installerer alle avhengigheter fra `requirements.txt`.

Appen apner normalt i nettleseren pa `http://localhost:8501`.

## Legg til dine data

Velg en av datakildene i appen:

- Eksporter Concept2 Logbook-data som CSV og legg filene i `data/system/Data/` med navn som `concept2-season-2026.csv`.
- Oppgi din egen read-only Concept2 access token i portalen for a hente data fra API-et. Tokenet lagres ikke av portalen.

Alle data lagres bare lokalt pa din maskin. Mappen `data/system/Data/` er tom i denne distribusjonen og er ignorert av Git, slik at personlige treningsdata, kommentarer, laktat, vekt, cache og token ikke skal bli publisert.

## Manuell oppstart

```powershell
cd data\system
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r treningshistorikk_app/requirements.txt
streamlit run treningshistorikk_app/app.py
```

## Personvern ved videreutvikling

F&#248;r du publiserer egne endringer, kontroller at `data/system/Data/`, `data/system/.streamlit/secrets.toml` og `data/system/logs/` ikke er med i commit-en. Bruk aldri en API-token i kildekode eller i filer som skal deles.