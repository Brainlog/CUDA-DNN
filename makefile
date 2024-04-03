sub3:
	nvcc -std=c++11 subtask3.cu -o subtask3
	./subtask3
sub4:
	nvcc -std=c++11 subtask4.cu -o subtask4
	./subtask4
setcuda:
	conda install cuda -c nvidia/label/cuda-11.8.0

