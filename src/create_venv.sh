
3DBB_VENV_DIR="${1:-./../3dbb_venv}"

REQUIREMENTS=requirements.txt

echo "******** Creating virtual environment ********"
python3.10 -m venv --system-site-packages "${3DBB_VENV_DIR}"


echo "******** Activating virtual environment ********"
source "${3DBB_VENV_DIR}/bin/activate"
pip3 install -U pip

pip3 install -r "${REQUIREMENTS}"