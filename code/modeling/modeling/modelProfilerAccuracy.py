#!python3

import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import tree
from sklearn.model_selection import train_test_split
from IPython.display import display

sns.set(style="darkgrid")


# HELPER Class
class Experiment:
    def __init__(self, config):
        self.config = config
        self.time = []

    def add_time(self, time):
        self.time.append(time)


class ModelProfilerAccuracy:
    def __init__(self):
        # Absolute Paths
        # pointing to catena bb measurements:
        self.catena_bb = '/run/media/max/6650AF2E50AF0441/measurement_results_Feb_19/catena/catena/'
        self.h2_bb = '/run/media/max/6650AF2E50AF0441/measurement_results_Feb_19/h2/h2/'
        self.sunflow_bb = '/run/media/max/6650AF2E50AF0441/measurement_results_Feb_19/sunflow/sunflow/'

        self.model_err_dst = '/run/media/max/6650AF2E50AF0441/processed_data_err_Feb_19/'

        # fine grained pkls
        self.catena_fine_grained_125_5_f = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/experiment_catena_125_5.pkl'
        self.catena_fine_grained_raw = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/catena/feature_pbd_125_5/kieker_args/'

        self.sunflow_fine_grained_125_5_f = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/experiment_sunflow_125_5.pkl'
        self.sunflow_fine_125_5_f_raw = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/sunflow/feature_pbd_125_5/kieker_args/'
        self.sunflow_fine_grained_49_7_f = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/experiment_sunflow_49_7.pkl'
        self.sunflow_fine_49_7_raw = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/sunflow/feature_pbd_49_7/kieker_args/'

        self.h2_fine_grained_125_5_f = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/experiment_h2_125_5.pkl'
        self.h2_fine_125_5_f_raw = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/h2_annotated/feature_pbd_125_5/kieker_args/'
        self.h2_fine_grained_125_5_t2 = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/experiment_h2_125_5_t2.pkl'
        self.h2_fine_125_5_t2_raw = \
            '/run/media/max/6650AF2E50AF0441/monitoring_results_alphaweb/h2_annotated/t_2_pbd_125_5/kieker_args/'

    def read_in_experiment(self, path):
        cfgs = os.listdir(path)
        experiment = {}

        for cfg in cfgs:
            cfg_path = os.path.join(path, cfg)
            if not os.path.isdir(cfg_path):
                continue
            perf = ModelProfilerAccuracy.read_repetitions(cfg_path)
            experiment[cfg] = perf

        df = pd.DataFrame(experiment, index=[0]).T
        df.reset_index(level=0, inplace=True)
        df.columns = ['config', 'perf']

        return df

    @staticmethod
    def read_repetitions(c_folder):

        monitoring_files = []
        for f in os.listdir(c_folder):
            # rm .png files in case of sunflow rendering engine
            if f.endswith('.png'):
                continue
            elif f.endswith('bb_time.txt'):
                continue
            elif f.endswith('out.txt'):
                continue
            elif f.endswith('err.txt'):
                continue
            monitoring_files.append(os.path.join(c_folder, f))

        perf = []
        for file in monitoring_files:
            f = open(file, "r")
            perf.append(float(f.readlines()[-1]))
        return np.mean(perf)

    def learn(self, dataframe, sub_sys, repetitions=5, df_column='perf'):

        errors = []
        for _ in range(repetitions):

            x_train, y_train, x_test, y_test = self.make_train_test_split(dataframe, y_identifier=df_column)
            x_train, x_test = self.transform_config(sub_sys, x_train, x_test)

            model = tree.DecisionTreeRegressor()

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            errors.append(self.prediction_error(y_test, y_pred))

        error = np.mean(errors)
        return error

    def make_train_test_split(self, df, percentage=0.2, x_identifier='config', y_identifier='perf'):
        train, test = train_test_split(df, test_size=percentage)

        x_train = train[x_identifier].values
        y_train = train[y_identifier].values
        x_test = test[x_identifier].values
        y_test = test[y_identifier].values

        return x_train, y_train, x_test, y_test

    def transform_config(self, sub_sys, x_train, x_test):
        # copied from Notebook "Coarse Grained PIM"
        if sub_sys == 'catena':
            x_train = [self.parse_config_catena(x) for x in x_train]
            x_test = [self.parse_config_catena(x) for x in x_test]

            x_train = [self.catena_config_transformation(x_tr) for x_tr in x_train]
            x_test = [self.catena_config_transformation(x_te) for x_te in x_test]

        elif sub_sys == 'h2':
            x_train = [self.parse_config_h2(x) for x in x_train]
            x_test = [self.parse_config_h2(x) for x in x_test]

        elif sub_sys == 'sunflow':
            x_train = [self.parse_config_sunflow(x) for x in x_train]
            x_test = [self.parse_config_sunflow(x) for x in x_test]

        else:
            print('no config transformation defined')
        return x_train, x_test

    def parse_config_catena(self, cfg):
        # cfg =  1_0_1_0_10_49_0_64_Butterfly-Full-adapted_6789ab_6789ab_6789ab_64
        # out = [1 0 1 0 10 49 0 64]
        return cfg.split('_')[:8]

    def catena_config_transformation(self, X):
        # translates HASH and GRAPH back into independent features
        # [hash, gamma, graph, phi, garlic, lambda, v_id, d]
        # [1.0, 0.0, 1.0, 0.0, 4.0, 49.0, 64.0, 192.0] ->
        # [1.0, 0.0, 1.0, 0.0, 4.0, 49.0, 64.0, 192.0]
        # print('before:',X)
        transformed_x = []

        for i, feature in enumerate(X):
            # print('i',i,'feature',feature, type(feature))
            if i == 0:
                if feature == '1':
                    transformed_x = transformed_x + [1.0, 0.0]
                else:
                    transformed_x = transformed_x + [0.0, 1.0]
            elif i == 2:
                if feature == '1':
                    transformed_x = transformed_x + [1.0, 0.0, 0.0, 0.0]
                elif feature == '2':
                    transformed_x = transformed_x + [0.0, 1.0, 0.0, 0.0]
                elif feature == '3':
                    transformed_x = transformed_x + [0.0, 0.0, 0.0, 1.0]
                elif feature == '4':
                    transformed_x = transformed_x + [0.0, 0.0, 1.0, 0.0]
            else:
                transformed_x.append(float(feature))
            # print(transformed_X)
        # print('after:',transformed_X)
        return transformed_x

    def parse_config_h2(self, cfg):
        # 2001_5001_0_0_0_0_1_0_0_0_0_0_0_0_0_0_50000
        # [2001, 5001, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return cfg.split('_')[:15]

    def parse_config_sunflow(self, cfg):
        # 1_21_0_0_0_0_128
        # [1, 21, 0, 0, 0, 0, 128]
        return cfg.split('_')[:5]

    def mape(self, truth, prediction):
        return np.mean(np.abs((truth - prediction) / truth))

    def mae(self, truth, prediction):
        return np.mean(np.abs(truth - prediction))

    def prediction_error(self, y_test, y_pred):
        # test for division by 0
        # if np.any(y_test == 0):
        #     print('MAE')
        #     error = mae(y_test, y_pred)
        # else:
        #     error = mae(y_test, y_pred)
        # return error
        return self.mae(y_test, y_pred)

    def coarse_grained_analyse(self, df, file):
        data = df.groupby(['m_name']).mean()
        data.reset_index(level=0, inplace=True)
        data.sort_values(by=['err'], inplace=True, ascending=False)

        # weighted stuff
        perf_sum_all_methods = data['perf'].sum()
        data['weight'] = data['perf'] / perf_sum_all_methods
        data['weighted_error'] = data['err'] * data['weight']

        return data['weighted_error'].mean()

    def plot_error(self, data, file):
        f, ax = plt.subplots(figsize=(20, 15))
        plt.xticks(rotation=90)
        ax.set(yscale="log")
        g = sns.barplot(x='m_name', y='weighted_error', data=data, ax=ax)
        # g.legend_.remove()
        g.set(xlabel='Method Names', ylabel='Error (MAE)', title='Error Per Method - ' + file)

    def plot_table(self, df):
        df.columns = ['name', 'bb', 'coarse', 'fine']
        df['overhead (bb-coarse)'] = None
        df['overhead (bb-fine)'] = None
        # print(df.head)
        df['color'] = ['catena', 'catena', 'catena', 'catena', 'h2', 'h2', 'sunflow', 'sunflow']
        df.set_index('name', inplace=True)

        df.replace(to_replace=[None], value=0, inplace=True)

        f, ax = plt.subplots(figsize=(10, 8))
        # plt.xticks(rotation=90)
        # ax.set(xscale="log")
        g = sns.scatterplot(x='coarse', y='bb', data=df, hue=df['color'])
        # g.legend_.remove()
        # g.set(xlabel='Method Names', ylabel='Error (MAE)', title='Error Per Method - ' + file)
        f.show()

    def model_bb_catena(self):
        # MAIN BB Model Accuracy
        # print('feature_pbd_125_5' : )
        bb_accs_catena = {}
        for sam_str in os.listdir(self.catena_bb):
            print(sam_str)

            sam_str_path = os.path.join(self.catena_bb, sam_str)
            for prof in os.listdir(sam_str_path):
                # print(prof)

                prof_path = os.path.join(sam_str_path, prof)
                df = self.read_in_experiment(prof_path)
                # print(np.mean(df['perf']), np.std(df['perf']))
                error = self.learn(df, 'catena')
                bb_accs_catena[sam_str] = error
                # break
            # break
        return bb_accs_catena

    def model_bb_h2(self):
        bb_accs_h2 = {}
        for sam_str in os.listdir(self.h2_bb):
            print(sam_str)
            if sam_str.startswith('t_2_pbd_125_5') or sam_str.startswith('t_2_pbd_49_7'):
                continue

            sam_str_path = os.path.join(self.h2_bb, sam_str)
            for prof in os.listdir(sam_str_path):
                # print(prof)

                prof_path = os.path.join(sam_str_path, prof)
                df = self.read_in_experiment(prof_path)
                # print(np.mean(df['perf']), np.std(df['perf']))
                # print(df)
                error = self.learn(df, 'h2')
                bb_accs_h2[sam_str] = error
                # break
            # break
        return bb_accs_h2

    def model_bb_sunflow(self):
        bb_accs_sunflow = {}
        for sam_str in os.listdir(self.sunflow_bb):
            print(sam_str)
            if sam_str.startswith('t_2_pbd_125_5') or sam_str.startswith('t_2_pbd_49_7'):
                continue

            sam_str_path = os.path.join(self.sunflow_bb, sam_str)
            for prof in os.listdir(sam_str_path):
                # print(prof)

                prof_path = os.path.join(sam_str_path, prof)
                df = self.read_in_experiment(prof_path)
                # print(np.mean(df['perf']), np.std(df['perf']))
                # print(df)
                error = self.learn(df, 'sunflow')
                bb_accs_sunflow[sam_str] = error
                # break
            # break
        return bb_accs_sunflow

    def model_coarse_grained(self):
        # coarse grained errors per method:
        coarse_accs = {}
        coarse_grained_pkls = os.listdir(self.model_err_dst)
        for file in coarse_grained_pkls:
            print(file)

            df = pd.read_pickle(os.path.join(self.model_err_dst, file))
            acc = self.coarse_grained_analyse(df, file)
            coarse_accs[file] = acc
        # print(coarse_accs)
        return coarse_accs

    def model_fine_grained(self, file, ss):
        # MAIN WB Accuracy
        print(file)
        wb_accs = {}
        df = pd.read_pickle(file)

        df = df.groupby(['name', 'config']).mean()
        df.reset_index(inplace=True)
        df_groups = df.groupby(['name'])
        print(len(df_groups))

        # ['name', 'config', 'length_o', 'mean_o', 'median_o', 'std_o', 'sum_o', 'length_r', 'mean_r',
        # 'median_r', 'std_r', 'sum_r', 'arg_el_net_err', 'arg_dec_tree_err', 'var_hist', 'rep']
        column_id = 'median_r'
        errors = {}
        for name, group in df_groups:
            print(name)
            print(list(group))
            out = self.learn(group, ss, df_column=column_id)
            # print(out.mean())
            print('error', out)
            per_per_cfg_df = group.groupby(['config']).mean()
            mean_perf = per_per_cfg_df[column_id].mean()
            print('mean perf', mean_perf)
            errors[name] = [out, mean_perf]

        print(errors)
        data = pd.DataFrame.from_dict(errors, orient='index', columns=['err', 'perf'])

        #display(df)
        perf_sum_all_methods = data['perf'].sum()
        data['weight'] = data['perf'] / perf_sum_all_methods
        data['weighted_error'] = data['err'] * data['weight']

        weighted_error = data['weighted_error'].mean()
        print('weighted error', weighted_error)

        return weighted_error

    def gen_err_df(self, bb_accs_catena, bb_accs_h2, bb_accs_sunflow, coarse_accs, fine_grained_errors):

        keys = list(coarse_accs.keys())

        bb_errors_catena = list(bb_accs_catena.values())
        bb_errors_h2 = list(bb_accs_h2.values())
        bb_errors_sunflow = list(bb_accs_sunflow.values())
        bb_errors = bb_errors_catena + bb_errors_h2 + bb_errors_sunflow

        coarse_errors = list(coarse_accs.values())
        fine_errors = fine_grained_errors

        to_df = [keys, bb_errors, coarse_errors, fine_errors]
        df = pd.DataFrame(to_df)
        return df.T

    # #####################################################################################

    def read_cfg_2(self, cfg):
        repetitions = []
        repetition_name = [run for run in os.listdir(cfg) if run.endswith('.pkl')]
        for i, rep in enumerate(repetition_name):
            df = pd.read_pickle(os.path.join(cfg, rep))
            save = [self.variance_of_hist_2(row['hist'], row['bin_edges']) for index, row in df[['hist', 'bin_edges']].iterrows()]
            df['var_hist'] = save
            df['rep'] = i
            repetitions.append(df)
        return pd.concat(repetitions)

    def variance_of_hist_2(self, hist, bins):
        # Calc Mean = Sum(i=1 to N=#OfBins) i*hist(i)
        mean_val = 0

        width = (bins[1] - bins[0]) / 2
        centered_bins = [single_bin + width for single_bin in bins[:-1]]
        total_n = sum(hist)
        bins_n = len(bins) - 1

        weighted_bins = [h * b for h, b in list(zip(hist, centered_bins))]
        total_p = sum(centered_bins)
        mean_val = total_p / bins_n

        # Calc Variance = Sum(i=1 to N=#OfBins) (i-Mean)^2 * hist(i)
        variance_arr = [math.pow(b - mean_val, 2) * h for h, b in list(zip(hist, centered_bins))]
        return sum(variance_arr) / total_n

    def read_in_experimant_data_2(self, path, pkl_path):
        if os.path.exists(pkl_path):
            return pd.read_pickle(pkl_path)
        else:
            catena_f_125_configs = [os.path.join(path, cfg) for cfg in os.listdir(path)]
            exp_list = []

            for cfg in catena_f_125_configs:
                curr_cfg = os.path.basename(os.path.normpath(cfg))
                config = self.read_cfg_2(cfg)
                config['config'] = curr_cfg
                exp_list.append(config)

            experiment = pd.concat(exp_list)
            experiment.to_pickle(pkl_path)
            return experiment

    def parse_config_2(self, cfg, is_catena):
        # if catena rm all of the wl:
        # print(cfg)
        if cfg.endswith('Butterfly-Full-adapted_6789ab_6789ab_6789ab_64'):
            tmp = cfg.split('_')[:-5]
        else:
            tmp = cfg.split('_')[:-1]
        return [float(x) for x in tmp]

    def catena_X_transformation_2(self, X):
        # add ROOT feature at the start
        # translates HASH and GRAPH back into independent features
        # [hash, gamma, graph, phi, garlic, lambda, v_id, d]
        # [1.0, 0.0, 1.0, 0.0, 4.0, 49.0, 64.0, 192.0] ->
        # [1.0, 0.0, 1.0, 0.0, 4.0, 49.0, 64.0, 192.0]

        transformed_X = [1.0]

        for i, feature in enumerate(X):
            if i == 0:
                if feature == 1.0:
                    transformed_X = transformed_X + [1.0, 0.0]
                else:
                    transformed_X = transformed_X + [0.0, 1.0]
            elif i == 2:
                if feature == 1.0:
                    transformed_X = transformed_X + [1.0, 0.0, 0.0, 0.0]
                elif feature == 2.0:
                    transformed_X = transformed_X + [0.0, 1.0, 0.0, 0.0]
                elif feature == 3.0:
                    transformed_X = transformed_X + [0.0, 0.0, 0.0, 1.0]
                elif feature == 4.0:
                    transformed_X = transformed_X + [0.0, 0.0, 1.0, 0.0]
            else:
                transformed_X.append(feature)

        return transformed_X

    def make_train_test_split_methods_2(self, df, is_catena, percentage=0.2):
        train, test = train_test_split(df, test_size=percentage)

        x_train = train.loc[:, 'config']
        y_train = train.loc[:, list(train)[-1]].values
        x_test = test.loc[:, 'config']
        y_test = test.loc[:, list(train)[-1]].values

        x_train = [self.parse_config_2(x, is_catena) for x in x_train.values]
        x_test = [self.parse_config_2(x, is_catena) for x in x_test.values]

        if is_catena:
            x_train = [self.catena_X_transformation_2(x_tr) for x_tr in x_train]
            x_test = [self.catena_X_transformation_2(x_te) for x_te in x_test]

        return x_train, y_train, x_test, y_test

    def create_pim_2(self, data, name, is_catena):
        errors = []
        x_train, y_train, x_test, y_test = self.make_train_test_split_methods_2(data, is_catena)

        current_model = tree.DecisionTreeRegressor()
        current_model.fit(x_train, y_train)
        y_pred = current_model.predict(x_test)
        # errors.append(mape(y_test, y_pred))
        errors = self.medape_2(y_test, y_pred)

        return errors

    def mape_2(self, truth, prediction):
        return np.mean(np.abs((truth - prediction) / truth) * 100)

    def medape_2(self, truth, prediction):
        return np.median(np.abs((truth - prediction) / truth) * 100)

    def filter_dataframe_2(self, experiment, perf_prop='MEAN'):
        model = experiment.set_index(['name', 'config'])
        model.sort_index(inplace=True)

        grouped_model = model.groupby(level=[0, 1])
        aggregated_groups = None
        # here is the important point of the approach
        # the model accuracy strongly depends on the chosen performance metric [mean, median, ...]
        if perf_prop == 'MEAN':
            aggregated_groups = grouped_model['mean_o'].agg([np.mean])
        elif perf_prop == 'MEDIAN':
            aggregated_groups = grouped_model['median_o'].agg([np.mean])

        return aggregated_groups

    def wb_fine_grained(self, path, pkl_file, is_catena=False):
        error = {}

        exp_catena_125_f = self.read_in_experimant_data_2(path, pkl_file)
        # print(list(exp_catena_125_f))
        aggregated_groups = self.filter_dataframe_2(exp_catena_125_f, 'MEDIAN')
        # print(aggregated_groups)
        # rm config as key and group by methodname
        aggregated_groups = aggregated_groups.reset_index(level='config')
        aggregated_groups = aggregated_groups.groupby('name')
        # print(aggregated_groups)
        for name, group in aggregated_groups:
            out = self.create_pim_2(group, name, is_catena)
            error[name] = [out, group['mean'].mean()]

        # calc weighted error over all methods
        df = pd.DataFrame(error).T
        df.columns = ['err', 'perf']

        perf_sum_all_methods = df['perf'].sum()
        df['weight'] = df['perf'] / perf_sum_all_methods
        df['weighted_error'] = df['err'] * df['weight']
        out = df['weighted_error'].sum()

        return out

    # #######################################################################################

    def model(self):

        print('BB')
        bb_accs_catena = self.model_bb_catena()
        bb_accs_h2 = self.model_bb_h2()
        bb_accs_sunflow = self.model_bb_sunflow()
        #
        print('WB-C')
        coarse_accs = self.model_coarse_grained()

        print('WB-F')
        # bla = self.model_fine_grained(file=self.catena_fine_grained_125_5_f, ss='catena')
        # print('error catena 125 5 f fine', bla)
        # print('error sunflow 125 5 f fine', self.model_fine_grained(file=self.sunflow_fine_grained_125_5_f, ss='sunflow'))
        # print('error sunflow 49 7 f fine', self.model_fine_grained(file=self.sunflow_fine_grained_49_7_f, ss='sunflow'))
        # # print('error h2 125 5 f fine', self.model_fine_grained(file=self.h2_fine_grained_125_5_f, ss='h2'))
        # print('error h2 125 5 t2 fine', self.model_fine_grained(file=self.h2_fine_grained_125_5_t2, ss='h2'))

        catena_fine_125_f_err = self.wb_fine_grained(self.catena_fine_grained_raw,
                                                     self.catena_fine_grained_125_5_f,
                                                     is_catena=True)

        sunflow_fine_125_f_err = self.wb_fine_grained(self.sunflow_fine_125_5_f_raw,
                                                      self.sunflow_fine_grained_125_5_f,
                                                      is_catena=False)
        sunflow_fine_49_f_err = self.wb_fine_grained(self.sunflow_fine_49_7_raw,
                                                     self.sunflow_fine_grained_49_7_f,
                                                     is_catena=False)

        h2_fine_125_f_err = self.wb_fine_grained(self.h2_fine_125_5_f_raw,
                                                 self.h2_fine_grained_125_5_f,
                                                 is_catena=False)
        h2_fine_125_t2_err = self.wb_fine_grained(self.h2_fine_125_5_t2_raw,
                                                  self.h2_fine_grained_125_5_t2,
                                                  is_catena=False)

        # print()
        # print()
        # print('Keys', list(coarse_accs))
        #
        # print('BB')
        # print('catena', bb_accs_catena)
        # print('h2', bb_accs_h2)
        # print('sunflow', bb_accs_sunflow)
        #
        # print('WB C')
        # print(coarse_accs)
        #
        # print('WB F')
        # print('catena', catena_fine_125_f_err)
        # print('h2', h2_fine_125_f_err, h2_fine_125_t2_err)
        # print('sunflow', sunflow_fine_125_f_err, sunflow_fine_49_f_err)

        fine_grained_errors = [catena_fine_125_f_err,
                               None,
                               None,
                               None,
                               h2_fine_125_f_err,
                               h2_fine_125_t2_err,
                               sunflow_fine_125_f_err,
                               sunflow_fine_49_f_err]

        err_df = self.gen_err_df(bb_accs_catena, bb_accs_h2, bb_accs_sunflow, coarse_accs, fine_grained_errors)
        err_df.columns = ['software system', 'BB', 'WB Coarse', 'WB Fine']
        display(err_df)
        self.plot_table(err_df)

        print('done.')


def main(args):

    print('start modeling')
    print('Number args:', len(args))
    comp = ModelProfilerAccuracy()
    comp.model()


if __name__ == "__main__":
    main(sys.argv)
