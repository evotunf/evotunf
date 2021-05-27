.PHONY: all install clean

.ONESHELL:
all: install
	python3 main_shuttle.py -i 150 -r 15 -p 100
valgrind: install
	valgrind \
		 --log-file=valgrind-out.txt \
		 python3 main_iris.py -i 0 -r 5 -p 10
install:
	cd evotunf && python3 setup.py build_ext --debug -j3 install --user
clean:
	pip3 uninstall -y evotunf
	rm -rf evotunf/build evotunf/dist evotunf/evotunf.egg-info
