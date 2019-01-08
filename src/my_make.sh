FILE="main"
if [ -e $FILE ];then
        echo "remove $FILE (old version)"
        rm main
fi

make