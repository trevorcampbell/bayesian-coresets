#!/bin/bash

#!/bin/bash

for ID in {1..3}
do
    for alg in "US" "SVI" "SVI-EXACT" "GIGA-REAL" "GIGA-REAL-EXACT" "GIGA-OPT" "GIGA-OPT-EXACT"
    do 
        python3 main.py --alg $alg --trial $ID run
    done
done
