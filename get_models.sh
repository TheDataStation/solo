if [ "$#" -ne 1 ]; then
  echo "Usage: ./get_models.sh <server>"
  exit
fi
work_dir="$(dirname "$PWD")"
server=$1
cd ${work_dir}
rm -f models.tar
wget http://${server}/open_table_discovery/models.tar
tar -xf models.tar
rm -f models.tar
