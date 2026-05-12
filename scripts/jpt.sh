
source ~/vqdata/bin/activate
echo "vqdata" activated
./j/kill_ports.sh

CUDA_VISIBLE_DEVICES=0 jupyter lab --port=8900 &
CUDA_VISIBLE_DEVICES=1 jupyter lab --port=8901 &
CUDA_VISIBLE_DEVICES=2 jupyter lab --port=8902 &
CUDA_VISIBLE_DEVICES=3 jupyter lab --port=8903 &

