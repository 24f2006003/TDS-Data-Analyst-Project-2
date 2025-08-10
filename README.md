# TDS-Data-Analyst-Project-2
This project provides a data analyst API for processing questions via HTTP requests.

## Usage

Health check:
```sh
curl -k "https://data-analyst-agent-tds.onrender.com/"
```

POST questions:
```sh
curl -k -L -X POST "https://data-analyst-agent-tds.onrender.com/api" -F "questions.txt=@question.txt"
```