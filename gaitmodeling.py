import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from statsmodels.formula.api import ols


bin_grps=["base_tail_lateral_displacement", "limb_duty_factor", "nose_lateral_displacement",
                  "speed_cm_per_sec", "step_length1", "step_length2", "step_width", "stride_length",
                  "temporal_symmetry", "tip_tail_lateral_displacement"]
bin_ang_vel_neg20=["speed_10_ang_vel_neg20", "speed_15_ang_vel_neg20",
                 "speed_20_ang_vel_neg20", "speed_25_ang_vel_neg20",
                 "speed_30_ang_vel_neg20", "speed_35_ang_vel_neg20",
                 "speed_40_ang_vel_neg20", "speed_45_ang_vel_neg20"]

class GaitData:
    def __init__(self, gait_h5_file, meta, bin_grps, grpname, output_dir,
                 outlier_threshold=5,
                 read_from_h5_file=True,
                 remove_outlier=True):

        gait_h5_basename = os.path.basename(gait_h5_file)

        self.gait_h5_file=gait_h5_file
        # self.meta_file=meta_file
        # self.meta=pd.read_csv(self.meta_file, sep="\t")
        self.meta=meta
        self.bin_grps=bin_grps
        self.grpname=grpname
        self.remove_outlier=remove_outlier
        self.outlier_threshold=outlier_threshold
        self.read_from_h5_file=read_from_h5_file
        self.prefix, _ = os.path.splitext(gait_h5_basename)
        self.tsv_file = os.path.join(output_dir, self.prefix + ".tsv")
        self.df=pd.DataFrame()
        self.df_merged=pd.DataFrame()
        self.df_cleaned=pd.DataFrame()
        self.z_score=0

        if gait_h5_basename.startswith("gait-mutants"):
            self.meta_cols_to_use = ["NetworkFilename", "MouseID", "Strain", "Mutant", "Sex", "TestAge"]
        elif gait_h5_basename.startswith("strain-survey"):
            self.meta_cols_to_use = ["NetworkFilename", "MouseID", "Strain", "Sex"]
        elif gait_h5_basename.startswith("autism-kos"):
            self.meta_cols_to_use = ["NetworkFilename", "MouseID", "Strain", "Sex", "TestAge"]
        elif gait_h5_basename.startswith("KOMP-curated"):
            raise NotImplementedError('need to implement KOMP')
        else:
            raise ValueError("h5 filename need start with gait-mutants, strain-survey or autism-kos")

    def get_h5_data(self, h5file):
        print("==========Reading H5 file:" + h5file + "==========")
        dfs = []
        with h5py.File(h5file, 'r') as gait_h5:
            for grp_name, items in gait_h5.items():
                for bin_ang_vel in self.grpname:
                    df_obj = pd.DataFrame(columns=["NetworkFilename","bingrpname","dim","BodyLength"] + self.bin_grps)
                    for grp in self.bin_grps:
                        path = "bins/" + bin_ang_vel + "/" + grp
                        if path in items:
                            arr = np.array(items["bins/" + bin_ang_vel + "/" + grp])
                            dim = len(arr)
                            df_obj[grp] = arr
                    df_obj["dim"] = dim
                    df_obj["bingrpname"] = bin_ang_vel
                    df_obj["NetworkFilename"] = grp_name
                    df_obj["BodyLength"] = items.attrs['median_body_length_cm']
                    dfs.append(df_obj)

        return pd.concat(dfs, ignore_index=True)

    def merge_pheno_data(self, df, meta,cols_to_use):
        print("==========Merge Phenotype data==========")
        meta["NetworkFilename"]=meta["NetworkFilename"].str.replace("/","%2F")
        cols_to_use=self.meta_cols_to_use
        if not set(cols_to_use).issubset(meta.columns):
            raise ValueError("meta file must contain: " + str(cols_to_use) +
                             " columns. Missing: " + str(set(cols_to_use) - set(meta.columns)))
        meta=meta[cols_to_use]
        meta=meta.drop_duplicates()
        merged=pd.merge(df, meta[cols_to_use], how="left", on="NetworkFilename")
        print(merged.shape)
        return merged

    def is_outlier(self, points,thresh):
        """
        Returns a boolean array with True if points are outliers and False
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The  z-score to use as a threshold. Observations with 
                 z-score (based on the median absolute deviation) bigger than the threshold will be classified as outliers.
        """

        if len(points.shape) == 1:
            points = points[:, None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median) ** 2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        self.z_score = diff / med_abs_deviation
        plt.plot(self.z_score, "o")
        plt.ylabel("Z-Score")
        plt.xlabel("observations")
        #thresh=np.quantile(z_score,1-self.outlier_threshold)
        return self.z_score > thresh

    def process_data(self):
        self.df=self.get_h5_data(self.gait_h5_file)
        print("Initial num of video groups:"+ str(len(self.df["NetworkFilename"].unique())))
        self.df_merged=self.merge_pheno_data(self.df,self.meta,self.meta_cols_to_use)
        self.df_merged=self.df_merged.dropna(subset=self.bin_grps)
        print("Num of video groups after removing NA:"+ str(len(self.df_merged["NetworkFilename"].unique())))
        if self.remove_outlier:
            df_outlier=pd.DataFrame(columns=self.bin_grps)
            groups=self.df_merged.groupby('NetworkFilename')
            for gait_param in self.bin_grps:
                print("==========Find outliers for ",gait_param,"==========")
                df_outlier[gait_param]=groups[gait_param].transform(lambda x: ~self.is_outlier(x,self.outlier_threshold))
            self.df_cleaned=self.df_merged.loc[df_outlier.all(axis=1)]
            print("Before removing outliers: ", self.df_merged.shape)
            print("After removing outliers: ", self.df_cleaned.shape)
            print("Removed ", self.df_merged.shape[0]-self.df_cleaned.shape[0], " points, "
                  "{0:.2%}".format((self.df_merged.shape[0]-self.df_cleaned.shape[0])*1.0/self.df_merged.shape[0]))
            print("Num of video groups after removing outliers:"+ str(len(self.df_cleaned["NetworkFilename"].unique())))
            return self.df_cleaned
        else:
            return self.df_merged


class Regression:
    def __init__(self,df,gait_params,formula_1="~BodyLength+speed_cm_per_sec", formula_2="",plot_residual=False, save_file=True):
        self.gait_params = gait_params
        self.df = df
        self.df_res=self.df.copy()
        self.df_coeff1=pd.DataFrame()
        self.df_coeff2=pd.DataFrame()
        self.df_aov1=pd.DataFrame()
        self.df_aov2=pd.DataFrame()
        self.formula_1=formula_1
        self.formula_2=formula_2
        self.plot_residual=plot_residual
        self.do_regression_2=True
        if formula_2=="":
            self.do_regression_2=False

    def ols_regression(self,df,gait_param,formula):
        model=ols(formula, data=df).fit()
        df["resid"]=model.resid
        if self.plot_residual:
            self.plot_resid(model,gait_param)
        aov=sm.stats.anova_lm(model, typ=2)
        aov["df"]=aov["df"].astype(int)
        aov["p_val(F;df)"]=aov["PR(>F)"].apply(lambda x:  '{0:1.2e}'.format(x)) +"(" +aov["F"].round(2).map(str) +";"+ aov["df"].round(2).map(str)+")"

        aov=aov.add_prefix(gait_param+'_')
        #print(aov)
        df_coeff=pd.DataFrame(model.params)
        df_coeff.index=df_coeff.index+"_coeff"
        df_se=pd.DataFrame(model.bse)
        df_se.index=df_se.index+"_se"
        df_p=pd.DataFrame(model.pvalues)
        df_p.index=df_p.index+"_p"
        return [df, pd.concat([df_coeff,df_se,df_p]).T, aov]

    def run(self):  
        for gait_param in self.gait_params:
            print("******regression for: "+gait_param+"******")
            formula_1 = gait_param + self.formula_1
            self.df_res,df_coeff1_temp, aov1=self.ols_regression(self.df_res,gait_param,formula_1)
            self.df_res.rename(columns={"resid":gait_param+"_resid"}, inplace=True)
            self.df_coeff1=pd.concat([self.df_coeff1, df_coeff1_temp], axis=0, ignore_index=True)
            self.df_aov1=pd.concat([self.df_aov1, aov1], axis=1)
            
            if self.do_regression_2:
                formula_2 = gait_param+"_resid" + self.formula_2
                self.df_res,df_coeff2_temp, aov2=self.ols_regression(self.df_res,gait_param,formula_2)
                self.df_res.rename(columns={"resid":gait_param+"_resid2"}, inplace=True) 
                self.df_aov2=pd.concat([self.df_aov2, aov2], axis=1)
                self.df_coeff2=pd.concat([self.df_coeff2,df_coeff2_temp],axis=0,ignore_index=True)
        self.df_coeff1.insert(loc=0, column='gait_param', value=self.gait_params)
        self.df_coeff2.insert(loc=0, column='gait_param', value=self.gait_params)
        return [self.df_res, self.df_coeff1, self.df_coeff2]
     
    def plot_resid(self, model,gait_param):
        f, axes = plt.subplots(1, 2, figsize=(8, 4))
        sns.distplot(model.resid,ax=axes[0])
        sm.qqplot(model.resid, line='r',ax=axes[1])
        plt.suptitle(gait_param, fontsize=10,y = 1.02)
        axes[0].set_xlabel("residual") 
        axes[0].set_ylabel("dist") 
        plt.tight_layout()


def write_resid_to_h5(h5file, df, gait_params):
    with h5py.File(h5file, 'w') as gait_h5:
        for grp_name in df["NetworkFilename"].unique():
            print("Creating:", grp_name)
            df_grp = df[df["NetworkFilename"] == grp_name]
            for bingrpname in df["bingrpname"].unique():
                df_bin = df_grp[df_grp["bingrpname"]==bingrpname]
                for gait_param in gait_params:
                    resid_name = gait_param + "_resid"
                    resid = df_bin[resid_name]
                    filtered_name = gait_param + "_filtered"
                    gait_h5.create_dataset(grp_name + "/bins/" + bingrpname + "/" + filtered_name, data=df_bin[gait_param])
                    gait_h5.create_dataset(grp_name + "/bins/" + bingrpname + "/" + resid_name, data=resid)


def write_resid_to_csv(output_dir, prefix,df_res, df_coeff1, df_coeff2, df_aov2=None, gait_params=None):
    resid_file = os.path.join(output_dir, prefix + "_resid.csv")
    coeff1_file = os.path.join(output_dir, prefix + "_coeff1.csv")
    coeff2_file = os.path.join(output_dir, prefix + "_coeff2.csv")
    aov_file = os.path.join(output_dir, prefix + "_aov2.csv")

    if df_aov2 is not None:
        cnames = [gait_param+"_p_val(F;df)" for gait_param in gait_params]
        temp = df_aov2[cnames].T
        temp.to_csv(aov_file, index=True, sep="\t")

    print("Writing file " + resid_file)
    df_res.to_csv(resid_file, index=False, sep="\t")
    print("Writing file " + coeff1_file)  
    df_coeff1.to_csv(coeff1_file, index=False, sep="\t")
    if df_coeff2.shape[1]>1:
        print("Writing file " + coeff2_file)
        df_coeff2.to_csv(coeff2_file, index=False, sep="\t")
