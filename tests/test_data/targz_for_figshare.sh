for dir in ./*/
do
    dir_for_tar=${dir%*/}
    dir_for_tar=${dir_for_tar#./}
    set -x
    cd ${dir}    
    tar --exclude=".[^/]*" -czvf ../${dir_for_tar}.tar.gz .
    cd ..
    set +x
done
