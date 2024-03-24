all:
	nvcc subtask3.cu -o subtask3
	./subtask3
setcuda:
	conda install cuda -c nvidia/label/cuda-11.8.0

