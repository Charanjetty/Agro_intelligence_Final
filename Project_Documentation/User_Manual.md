# 📘 AgroIntelligence 2.0 - Beginner's User Manual

Welcome! This guide will help you set up and run the AgroIntelligence project on your own computer, step-by-step. No advanced coding knowledge is required.

## Prerequisites
Before starting, ensure you have:
1. **Python** installed (Version 3.8 or higher). [Download Here](https://www.python.org/downloads/)
2. **Git** installed. [Download Here](https://git-scm.com/downloads)

---

## 🚀 Installation Steps

### Step 1: Clone the Repository
Open your command prompt (terminal) and run the following command to download the project code:

```bash
git clone https://github.com/Charanjetty/Agro_intelligence_Final.git
cd Agro_intelligence_Final

```

### Step 2: Create a Virtual Environment
It's best to verify your Python installation first:
```bash
python --version
```
Now, create a clean environment for the project to avoid conflicts:
```bash
# Windows
python -m venv .venv

# Mac/Linux
python3 -m venv .venv
```

### Step 3: Activate the Environment
You need to turn on the environment you just created:

- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source .venv/bin/activate
  ```
*(You should see `(.venv)` appear at the start of your command line.)*

### Step 4: Install Dependencies
Install all the necessary tools and libraries the project needs:
```bash
pip install -r requirements.txt
```
*Note: This might take a few minutes as it downloads tools like TensorFlow, Flask, and Pandas.*

---

## 🏃 Running the Application

### Step 5: Start the Server
Once installation is complete, you can start the website:

```bash
python app.py
```

### Step 6: Access the Website
Open your web browser (Chrome, Edge, Firefox) and go to:
[http://localhost:5000](http://localhost:5000)

Congratulations! 🎉 You actived AgroIntelligence 2.0 locally.

---

## ❓ Troubleshooting

- **"Python is not recognized"**: Ensure you checked "Add Python to PATH" during installation.
- **"No Python at..." error**: This happens if the Python version on your system changed or the `.venv` was moved. To fix it:
  1. Delete the `.venv` folder.
  2. Run `python -m venv .venv` again.
  3. Re-activate and run `pip install -r requirements.txt`.
- **"Module not found"**: Make sure your virtual environment is activated (`(.venv)` is visible) and you ran `pip install -r requirements.txt`.
