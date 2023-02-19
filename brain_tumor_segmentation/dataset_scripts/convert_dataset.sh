#!/bin/bash

# taken from https://stackoverflow.com/q/59895/11913195
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PY_SCRIPT="$SCRIPT_DIR/process_dataset.py"

stop_prog () {
    echo "Usage: $0 [ -d DATASET_PATH ]" 1>&2
    exit 1
}
# get dataset from user
while getopts d: OPTION
do
    case "$OPTION" in
        d) DATASET=${OPTARG};;
    esac
done

if [ -z "$DATASET" ]; then
    stop_prog
fi

# for each image in the dataset process them
LBC='\033[1;34m'
NC='\033[0m'
process_image3d () {
    printf "${LBC}Starting transform 3D${NC}\n"
    python "${PY_SCRIPT}" --path2image "${1}" --label_brain "${2}"\
        --label_tumour "${3}"
}
process_image4d () {
    printf "${LBC}Starting transform 4D${NC}\n"
    python "${PY_SCRIPT}" --path2image "${1}" --label_brain "${2}"\
        --label_tumour "${3}" --tumour_0th_axis_idx "${4}"\
        --brain_0th_axis_idx "${5}"
}

# process 3d images
process_image3d "${DATASET}/ahmet_timur_label/Segmentation.seg.nrrd"\
    1 2
process_image3d "${DATASET}/enver_akkaya_label/Segmentation.seg.nrrd"\
    2 1
process_image3d "${DATASET}/munevver_altan_label/Segmentation.seg.nrrd"\
    1 2
process_image3d "${DATASET}/osman_altintas/Segmentation.seg.nrrd"\
    2 1
process_image3d "${DATASET}/ramazan_acer_label/Segmentation.seg.nrrd"\
    1 2
process_image3d "${DATASET}/salih_adali_label/Segmentation.seg.nrrd"\
    2 3
process_image3d "${DATASET}/zeynep_altınisik_label/Segmentation.seg.nrrd"\
    2 1

# process 4d images
process_image4d "${DATASET}/altun_nurten_label/Segmentation.seg.nrrd"\
    1 1 0 1
process_image4d "${DATASET}/cansel_akgun_label/Segmentation.seg.nrrd"\
    2 1 1 0
process_image4d "${DATASET}/elif_ece_altun_label/Segmentation.seg.nrrd"\
    1 1 1 0
