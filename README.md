<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 👥 Team

This project is developed by:

| Name | GitHub Username | Email |
|------|----------------|-------|
| Mamoun Jamai | [@mamounjamai](https://github.com/mamounjamai) | mamoun.jamai@hec.edu |
| Mehdi Digua | [@MehdiD19](https://github.com/MehdiD19) | mehdi.digua@hec.edu |
| Pierre Lafarguette | [@plafarguette2](https://github.com/plafarguette2) | pierre.lafarguette@hec.edu |
| Marco Salerno | [@sqerbo01](https://github.com/sqerbo01) | marco.salerno@hec.edu |
| Iliass Sijelmassi | [@iliassSjm](https://github.com/iliassSjm) | iliass.sijelmassi@hec.edu |
| Cedric Kire | [@cedrickirek](https://github.com/cedrickirek) | cedric.kire@hec.edu |

---

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Clone your forked repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ML-Ops-Project.git
   cd ML-Ops-Project
   ```

4. **Set up your development environment**:
   ```bash
   # Sync dependencies and create virtual environment
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate  # macOS/Linux
   # OR on Windows: .venv\Scripts\activate

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

5. **Verify your setup**:
   ```bash
   # Check Python version
   python --version  # Should be 3.11.x

   # Test pre-commit hooks
   uv run pre-commit run --all-files
   ```

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_1)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_1)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 🛠️ Development Workflow

### Running Code Quality Checks

```bash
# Run linter
uv run ruff check .

# Run linter with auto-fix
uv run ruff check . --fix

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

### Managing Dependencies

```bash
# Add a new package
uv add package-name==version

# Add a development package
uv add --dev package-name==version

# Sync environment after changes
uv sync

# Update lock file
uv lock
```

### Git Workflow for Each Branch

```bash
# 1. Switch to the branch
git checkout branch_name

# 2. Pull latest changes from main (except for branch_1)
git pull origin main

# 3. Create your working branch
git checkout -b branch_name-your-name

# 4. Make your changes, then commit
git add .
git commit -m "descriptive message"

# 5. Push your branch
git push -u origin branch_name-your-name

# 6. Create Pull Request on GitHub
# Merge to main after review
```

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

## 🤖 **Running the ML Scripts**

### Quick Start Training & Prediction

```bash
# 1. Train a model (downloads dataset automatically)
python -m src.modelling.main abalone.csv

# 2. Train with different model type
python -m src.modelling.main abalone.csv --model_type random_forest

# 3. View experiment results
mlflow ui  # Open http://localhost:5000
```

### What Each Script Does

- **`main.py`**: Entry point that orchestrates the complete training pipeline
- **`preprocessing.py`**: Data loading, feature engineering, and train/test splitting
- **`training.py`**: Model training with MLflow tracking, saves pickle files for deployment
- **`predicting.py`**: Load trained models and make predictions on new data
- **`utils.py`**: Helper functions for pickle serialization and data download

### Script Workflow
1. **Data**: Downloads abalone dataset if not found
2. **Preprocessing**: Feature engineering (ratios, log transforms, scaling)
3. **Training**: Fits model, logs metrics to MLflow, saves pickle files
4. **Deployment Ready**: Model and scaler saved for web service predictions

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
