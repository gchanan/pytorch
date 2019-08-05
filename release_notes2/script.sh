
while read p <&3; do
    grep "$p" processed.txt > /dev/null
   if [ $? -eq 0 ]; then
       echo "Skipping " $p
       continue
    fi
    git log -s -n 1 "$p"
    echo -e 'JIT (j)  \tVisualization (v)\tBC (c)  \tFeature (f)\tImprovement (i)\t\tBug Fix (b)'
    echo -e 'Deprecation (r)\tPerformance (p)   \tDocs (d)\tOnnx (o)\tSkip (s)\tCaffe2 (2)'
    echo -e 'Distributed (t)\tBuild(u)\tNot Important(n)\tQuantization (q)\tDistributions (w)\tAMD (a)'
    read -p "Enter option: " option < /dev/stdin
    echo "option " "$option"
    if [ "$option" == "j" ]; then
	echo "$p" >> results/jit.txt
    elif [ "$option" == "v" ]; then
	echo "$p" >> results/visualization.txt
    elif [ "$option" == "c" ]; then
	echo "$p" >> results/compat.txt
    elif [ "$option" == "f" ]; then
	echo "$p" >> results/feature.txt
    elif [ "$option" == "i" ]; then
	echo "$p" >> results/improvement.txt
    elif [ "$option" == "b" ]; then
	echo "$p" >> results/bug.txt
    elif [ "$option" == "r" ]; then
	echo "$p" >> results/deprecation.txt
    elif [ "$option" == "p" ]; then
	echo "$p" >> results/performance.txt
    elif [ "$option" == "d" ]; then
	echo "$p" >> results/docs.txt
    elif [ "$option" == "o" ]; then
	echo "$p" >> results/onnx.txt
    elif [ "$option" == "s" ]; then
	echo "$p" >> results/skip.txt
    elif [ "$option" == "2" ]; then
	echo "$p" >> results/caffe2.txt
    elif [ "$option" == "t" ]; then
	echo "$p" >> results/distributed.txt
    elif [ "$option" == "u" ]; then
	echo "$p" >> results/build.txt
    elif [ "$option" == "n" ]; then
    	echo "$p" >> results/not.txt
    elif [ "$option" == "q" ]; then
	echo "$p" >> results/quantization.txt
    elif [ "$option" == "w" ]; then
	echo "$p" >> results/distributions.txt
    elif [ "$option" == "a" ]; then
	echo "$p" >> results/amd.txt
    else
	echo "$p" >> results/wrong.txt
	exit 1
    fi
	echo "$p" >> processed.txt
done 3< commit_list2.txt

# bug.txt  compat.txt  deprecation.txt  docs.txt  feature.txt  improvement.txt  jit.txt  onnx.txt  performance.txt  skip.txt  visualization.txt caffe2.txt
