% Example run
clear all;
close all;


GTDIR = 'ENTER GROUND-TRUTH DIRECTORY';
ALG_DIR(1) = candidateAlgStructure('ALGORITHM A',A_Dir,[],'_Saliency','png');
ALG_DIR(2) = candidateAlgStructure('ALGORITHM B',B_Dir,[],[],'png');
ALG_DIR(3) = candidateAlgStructure('ALGORITHM C',C_Dir,'Sal_',[],'png');
Dataset = datasetStructure('MSRA',GTDIR);
[HitRate , FalseAlarm,AUC] = performBenchmark(Dataset,ALG_DIR);

