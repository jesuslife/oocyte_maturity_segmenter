@echo off
REM --- Modify the script paths accordingly ---
call C:\Users\vladi\Miniconda3\condabin\activate.bat
call conda deactivate
call conda activate tracking
cd C:\ConceivableProjects\oocyte_maturity_segmenter
python predict.py --input input.jpg --output result.txt
exit