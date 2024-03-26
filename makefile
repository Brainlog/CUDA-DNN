sub3:
	nvcc subtask3.cu -o subtask3
	./subtask3
sub4:
	nvcc subtask4.cu -o subtask4
	./subtask4
setcuda:
	conda install cuda -c nvidia/label/cuda-11.8.0

