run:
	rm -rf main.py
	jupyter nbconvert main.ipynb --to python
	nohup python main.py &
	tail -f nohup.out