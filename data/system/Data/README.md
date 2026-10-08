# Local data directory

This folder is intentionally empty in the GitHub version of the project.

The application creates local metadata files here when you add comments, tags, lactate values, session weights, or settings. These files are ignored by Git.

To load training history, either:

- place Concept2 Logbook CSV exports named `concept2-season-YYYY.csv` in this folder and choose CSV as the data source; or
- enter your personal read-only Concept2 access token in the launcher or set `CONCEPT2_ACCESS_TOKEN` before starting the app.

Never commit CSV exports, API caches, tokens, or other files containing personal training data.