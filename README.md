## Generate Metapath and Input Pairs
Generate the input pairs by `generate_walk.py` using the following command.
```
$ python src/generate_walk.py [name of dataset] [length] [num_walk] [window_size]
```

_Parameters_:
* `name of dataset`: Similar to above.
* `length`: the length of the generated walks.
* `num_walk`: the number of times each node is covered by walks.
* `window_size`: the size of Skip-gram sampling window.


## Run
Run by the following command:
```
$ ./run.sh 
```
But before running `run.sh`, please look into it and tweak the settings.

_Parameter_:
* `ID`: the identifier of a certain training/testing, will be used in output file name.
You would see the performance in `./performance/`.
