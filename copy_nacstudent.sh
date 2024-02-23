SOURCE_DIR="./log_dir/m2mKD_nac_"
TARGET_DIR="./nacs/students"
mkdir -p $TARGET_DIR
cp "$SOURCE_DIR"0/finished_checkpoint.pth "$TARGET_DIR"/ReadInStudent.pth
for IDX in 1 2 3 4 5 6 7 8
do
    cp "$SOURCE_DIR""$IDX"/finished_checkpoint.pth "$TARGET_DIR"/MediatorStudent_"$IDX".pth
done
cp "$SOURCE_DIR"9/finished_checkpoint.pth "$TARGET_DIR"/ReadOutStudent.pth