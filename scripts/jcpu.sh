
source ~/vqdata/bin/activate
echo "vqdata" activated

CUDA_VISIBLE_DEVICES=-1 jupyter lab --port=8910 &

