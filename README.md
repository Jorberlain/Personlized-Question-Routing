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
$ python src/main.py --dataset $2 \
        --window-size 4 --neg-ratio 3 --embedding-dim 256 \
        --lstm-layers 1 --epoch-number 100 --batch-size 200 \
        --learning-rate 0.003 --cnn-channel 64 --lambda 100000 \
        --length $5 --coverage $4 \
        --precision_at_K 5 --id $3 --test-ratio 0.4 \
        #--include-content
        #--gen-metapaths --length 15 --coverage 3 --alpha 0.0 --metapaths "AQRQA" 
        #--preprocess --test-threshold 3 --proportion-test 0.1 --test-size 20\
```

_Parameter_:
* `ID`: the identifier of a certain training/testing, will be used in output file name.
You would see the performance in `./performance/`.
