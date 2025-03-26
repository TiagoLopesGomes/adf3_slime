#!bash

for i in *_preds*;
do
		python create_combined_fasta.py --json-dir ./$i --output ./$i/slide_window_peptides.fasta
        mafft --auto --addfragments ./$i/slide_window_peptides.fasta --reorder --thread -1 ./ctir.fasta > ./$i/binder_peptides_alignment.fasta
		pymsaviz -i ./$i/binder_peptides_alignment.fasta -o ./$i/binder_peptides_alignment.png --color_scheme Clustal
done
