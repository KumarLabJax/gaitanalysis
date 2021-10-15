
Joining the datasources:

    tssql 'select MRG.* from `MergedMetaList_2018-09-25.csv` as MRG inner join `20180827_Gait_Mutants.csv` as MUT on MRG.Cage = MUT.CageID' > metamut.csv
    tssql 'select MRG.* from `MergedMetaList_2018-12-19.csv` as MRG inner join `20181218_Gait Mutants.csv` as MUT on MRG.Cage = MUT.CageID' > metamut-2018-12-20.csv
    tssql 'select MUT.*, MRG.* from `MergedMetaList_2019-01-10.csv` as MRG inner join `20181218_Gait Mutants.csv` as MUT on MRG.Cage = MUT.CageID' > metamut-2019-01-15.csv

I then just copy-paste NetworkFilename values into `gait-mutants-2019-01-15.txt`

add 10 - 20 ns and b6j for same time period of test. Sean can generate this for us

Example of genpervidstats usage

    python genpervidstats.py --batch-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10_exists.txt' --root-dir '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10' --merge-with-csv data/metadata/metamutalldata.csv

Doing `vidplot` stuff:

    python vidplot.py --video-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151.avi'  --data-file '/media/sheppk/TOSHIBA EXT/gait-mutants-2018-10-10/LL1-B2B/2018-09-04_SPD/LL1-1_003890-M-AX11-5.28571428571429-43291-1-K1808151_pose_est.h5'

Getting sample counts for the strain survey:

    namecolumns strain-survey.csv | tssql 'select Strain, Sex, count(*) from `data/metadata/MergedMetaList_2018-11-27.csv` inner join `-` on col1 = NetworkFilename group by Strain, Sex' | csvtopretty - > strain-survey-sample-counts.txt

Getting all NetworkFilename for strain survey and f2

    tssql 'select if_then_else(SS.NetworkFilename <> "", SS.NetworkFilename, F2.NetworkFilename) as NetworkFilename from `StrainSurvey_Output_2018-08-07.csv` as SS outer join `B6129SF2_NetworkName2GenoID.csv` as F2 on SS.NetworkFilename = F2.NetworkFilename' > strain-survey-and-f2-filenames.csv

Extract only the rows we care about:

    tssql 'select MM.* from `MergedMetaList_2018-12-04.csv` as MM inner join `strain-survey-and-f2-filenames.csv` as SSF2 on MM.NetworkFilename = SSF2.NetworkFilename'

We now see:

    $ wc strain-survey-and-f2-metadata.csv strain-survey-and-f2-filenames.csv StrainSurvey_Output_2018-08-07.csv B6129SF2_NetworkName2GenoID.csv 
    2792   40916 1674107    strain-survey-and-f2-metadata.csv
    2792   2792  153705     strain-survey-and-f2-filenames.csv
    2430   2539  714842     StrainSurvey_Output_2018-08-07.csv
    363    363   33453      B6129SF2_NetworkName2GenoID.csv
    8377   46610 2576107    total

To get our final batch file we can do:

    tssql 'select NetworkFilename from `metamut-2018-12-20.csv` where networkPointsFound="FALSE"' | sed 1d > gait-batch-2018-12-20.txt

In order to get the `*-with-extra-ctrls-*` file I found 8 male and 8 female samples from the main metdata sheet that looked like they should work and imported them manualy.

# Video clips

    ffmpeg -i LL1-1_C57BL6J.avi -ss 520 -t 2 LL1-1_C57BL6J-clip.avi
    ffmpeg -i LL1-1_NOR_M.avi -ss 124 -t 2 LL1-1_NOR_M-clip.avi
