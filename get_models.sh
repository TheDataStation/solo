work_dir="$(dirname "$PWD")"
cd ${work_dir}
rm -f models.tar
wget http://plover.cs.uchicago.edu/open_table_discovery/models.tar
tar -xf models.tar
rm -f models.tar
