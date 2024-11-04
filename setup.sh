# Install kotlin compiler for benchmark
apt install -y kotlin

# download mxeval repo
git clone --depth=1 git@github.com:amazon-science/mxeval.git temp_repo
cp -r temp_repo/data ./data
cp -r temp_repo/mxeval ./mxeval
rm -rf temp_repo

# Install all requirements
pip install -r requirements.txt