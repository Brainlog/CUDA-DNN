sub3:
	nvcc -std=c++11 subtask3.cu -o subtask3
	./subtask3
sub4:
	nvcc -std=c++11 subtask4.cu -o subtask4
	./subtask4 1
setcuda:
	conda install cuda -c nvidia/label/cuda-11.8.0

clean: 
	rm subtask3 subtask4 subtask2

sub2:
	nvcc -std=c++11 subtask2.cu -o subtask2
	./subtask2 2 1 3 1 2 -1 3 4 5 -10 0 -7


