# Clone the  repository and then execute:
python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

# Register Kernel
python -m ipykernel install --user --name ml-practice --display-name "Python (ML Practice)"