# Sentiment Analysis Project

This project implements a **Sentiment Analysis system** using Python, FastAPI, and machine learning models.  
It includes training scripts, model persistence, and an API server for serving predictions.

<br/>

## 📂 Project Structure
```
SENTIMENT-ANALYSIS/
│
├── app/                       # Main application package
│   ├── __pycache__/           # Python cache files (auto-generated)
│   ├── config.py              # Configurations (seeds, constants, device setup)
│   ├── main.py                # Entry point for FastAPI/Flask app
│   ├── model_io.py            # Model save/load utilities
│   ├── model_train.py         # Training pipeline (preprocessing + training)
│   ├── utils.py               # Helper functions
│
├── models/                    # Trained model files (saved .pt / .pkl)
│
├── venv/                      # Virtual environment (not to be committed to Git)
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   ├── share/
│   └── pyvenv.cfg
│
├── README.md                  # Documentation for the project
├── requirements.txt           # Python dependencies
├── run_server.py              # Script to run API server


```
<br/>

## 🛠 Installation & Local Development

### 1. Prerequisites
```bash
- Python 3.12.10
- pip (Python package manager)
```
### 2. Clone the repository
```bash
git clone https://github.com/debbrath/Sentiment-Analysis.git
cd Sentiment-Analysis
```
### Step 3: Open VSCode
- Launch VSCode.
- Open your project folder 

### Step 4: Select the Interpreter
1.	Press Ctrl+Shift+P → type Python: Select Interpreter → Enter.
python -m venv venv
![Screenshot](https://github.com/debbrath/Sentiment-Analysis/blob/main/images/image1.png)

### 5. Create and activate a virtual environment
```bash
# On Windows PowerShell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate

On Linux/Mac
python -m venv env
source env/bin/activate


Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
cd F:\Python\Sentiment-Analysis
.\.venv\Scripts\Activate.ps1
 
 
python -m pip install --upgrade pip setuptools wheel --use-pep517

```
### 6. Install dependencies
```bash
pip install -r requirements.txt
```
### 7. Train the model (if not already trained)
```bash
(venv) PS F:\Python\Sentiment-Analysis> python -m app.model_train 
```
### 8. Run locally
```bash
(venv) PS F:\Python\Sentiment-Analysis> uvicorn app.main:app –reload
```
<br/>


## 🛠 Technologies Used

Python 3.9+

FastAPI – API framework

PyTorch / Scikit-learn – Model training

Pandas / NumPy – Data processing


<br/>

---

✍️ Author

Debbrath Debnath

📫 [Connect on LinkedIn](https://www.linkedin.com/in/debbrathdebnath/)

🌐 [GitHub Profile](https://github.com/debbrath) 
