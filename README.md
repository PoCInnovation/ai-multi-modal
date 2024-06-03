# ai-multi-modal

## Installation
```
python -m venv venv
pip install -r requirements.txt
```
## Add .env file with the following content:
```
API_KEY=your_api_key
```

## Run the app
```
fastapi run Models/Model_selection.py --port 8000

fastapi run Models/GPT.py --port 8001


```