work_dir="$(dirname "$PWD")"
cd ${work_dir}
rm -f models.tar
wget https://storage.googleapis.com/open_data_123/models.tar
tar -xf models.tar
rm -f models.tar
