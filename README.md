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

fastapi run Models/Diffusion.py --port 8002

fastapi run Models/Model_verification.py --port 8003

```

## Run APIs (dev)
```
fastapi dev Models/Model_selection.py --port 8000 --reload &
fastapi dev Models/GPT.py --port 8001 --reload &
fastapi dev Models/Diffusion.py --port 8002 --reload &
fastapi dev Models/Model_verification.py --port 8003 --reload &
fastapi dev Models/Model_GPT2.py --port 8004 --reload &
```