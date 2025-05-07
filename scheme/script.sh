#!/bin/bash
function create_xcf {
    color=$1
    taxon=$2
    image_size=$3
    threshold=$4
    classes=$5
    dataset=$6
	DIR=/home/xandao/Documentos/GitHub/dataset_gimp/${dataset}/imagens/${color}/${taxon}/${image_size}/${threshold}/f${classes}/
    DIR_XCF=/home/xandao/Documentos/GitHub/dataset_gimp/${dataset}/imagens/${color}/${taxon}/${image_size}/${threshold}/f${classes}/xcf       
    mkdir -p $DIR_XCF
    DIR_JPEG=/home/xandao/Documentos/GitHub/dataset_gimp/${dataset}/imagens/${color}/${taxon}/${image_size}/${threshold}/f${classes}/jpeg        
    echo $DIR
    mkdir -p $DIR_JPEG
    gimp -i -c -b "(xcf \"${DIR}\")" -b "(gimp-quit 0)"
    mv "${DIR}"*.xcf "${DIR_XCF}" && mv "${DIR}"*.jpeg "${DIR_JPEG}"
}

function convert_rescale {
    color=$1
    taxon=$2
    image_size=$3
    threshold=$4
    classes=$5
    dataset=$6
    DIR_XCF=/home/xandao/Documentos/GitHub/dataset_gimp/${dataset}/imagens/${color}/${taxon}/${image_size}/${threshold}/f${classes}/xcf/
    DIR_JPEG=/home/xandao/Documentos/GitHub/dataset_gimp/${dataset}/imagens/${color}/${taxon}/${image_size}/${threshold}/f${classes}/jpeg/
    echo $DIR_XCF
    if [ ${color} = "grayscale" ]
    then
        echo "GRAYSCALE"
        gimp -i -c -b "(rescale \"${DIR_XCF}\" $image_size 1)" -b "(gimp-quit 0)"
    else
        echo "RGB"
        gimp -i -c -b "(rescale \"${DIR_XCF}\" $image_size 0)" -b "(gimp-quit 0)"
    fi
    cd $DIR_XCF && mv *.jpeg $DIR_JPEG
}

function run_test {
    classes=$1
    taxon=$2
    threshold=$3
    dataset=$4
    for image_size in 256 400 512; do
        for color in RGB; do
            for j in `seq 1 ${classes}`;
            do
                create_xcf ${color} ${taxon} ${image_size} ${threshold} $j ${dataset}
            done
            for j in `seq 1 ${classes}`;
            do
                convert_rescale ${color} ${taxon} ${image_size} ${threshold} $j ${dataset}
            done
        done    
    done
}

# run_test 20 specific_epithet 20 imagens_george
# run_test 34 specific_epithet 10 imagens_george
# run_test 55 specific_epithet 5 imagens_george
# run_test 2 genus 2 imagens_george
# run_test 20 specific_epithet 20 imagens_br
# run_test 34 specific_epithet 10 imagens_br
# run_test 55 specific_epithet 5 imagens_br
