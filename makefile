
brandes:
	nvcc -Xptxas -O3 -arch=sm_60 brandes.cu -o brandes --std=c++11 -O3


clean:
	rm -f brandes 