import numpy as np
import config
import pandas as pd
import json

#####################################################

class Decision_optimization_model:
    def __init__(self,wz):
        with open(config.SM_save_file, "r") as r_file:
            data = json.load(r_file)
        archive_total = data['archive_total']
        self.archive_total = archive_total
        self.wz = wz

    def normalization(self):
        archive_total_arry = np.array(self.archive_total)
        norm = np.zeros((archive_total_arry.shape[0], archive_total_arry.shape[1]))
        for i in range(archive_total_arry.shape[0]):
            for j in range(archive_total_arry.shape[1]):
                norm[i, j] = (archive_total_arry[i, j] - min(archive_total_arry[:, j])) / (
                            max(archive_total_arry[:, j]) - min(archive_total_arry[:, j]))
        return norm

    def weights(self,norm):
        mid_norm = np.zeros((norm.shape[0], norm.shape[1]))
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                mid_norm[i, j] = norm[i, j] / sum(norm[:, j])
        ln_norm = np.zeros((norm.shape[0], norm.shape[1]))
        for n_i in range(norm.shape[0]):
            for n_j in range(norm.shape[1]):
                if norm[n_i, n_j] == 0:
                    ln_norm[n_i, n_j] = 0
                else:
                    #####################################################
                    ln_norm[n_i, n_j] = np.log(mid_norm[n_i, n_j])
        #####################################################
        norm_ln_norm_contact = np.zeros((norm.shape[0], norm.shape[1]))
        for c_i in range(norm.shape[0]):
            for c_j in range(norm.shape[1]):
                #####################################################
                norm_ln_norm_contact[c_i, c_j] = mid_norm[c_i, c_j] * ln_norm[c_i, c_j]
                #####################################################
        hj = np.zeros(norm.shape[1])
        for h_j in range(norm.shape[1]):
            hj[h_j] = -sum(norm_ln_norm_contact[:, h_j]) / np.log(len(norm_ln_norm_contact[:, h_j]))
        hs = np.average(hj)
        mid_hj_1 = np.zeros(norm.shape[1])
        mid_hj_2 = np.zeros(norm.shape[1])
        for m_h_j in range(norm.shape[1]):
            mid_hj_1[m_h_j] = 1 + hs - hj[m_h_j]
            mid_hj_2[m_h_j] = 1 - hj[m_h_j]
        whsj = np.zeros(norm.shape[1])
        whkj = np.zeros(norm.shape[1])
        for w_h_j in range(norm.shape[1]):
            whsj[w_h_j] = mid_hj_1[w_h_j] / sum(mid_hj_1)
            whkj[w_h_j] = mid_hj_2[w_h_j] / sum(mid_hj_2)
        whj = np.zeros(norm.shape[1])
        for w_j in range(norm.shape[1]):
            if hj[w_j] == 1:
                whj[w_j] = 0
            else:
                whj[w_j] = hs * whsj[w_j] + (1 - hs) * whkj[w_j]
        wz_arry = np.array(self.wz)
        mid_w = np.zeros(norm.shape[1])
        if all(whj == 0):
            mid_w = wz_arry
        else:
            for m_t_j in range(norm.shape[1]):
                mid_w[m_t_j] = np.sqrt(wz_arry[m_t_j] * whj[m_t_j])
        w = np.zeros(norm.shape[1])
        for t_j in range(norm.shape[1]):
            w[t_j] = mid_w[t_j] / sum(mid_w)
        return w

    def topsis(self,norm, w):
        weighting_norm = np.zeros((norm.shape[0], norm.shape[1]))
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                weighting_norm[i, j] = norm[i, j] * w[j]
        best_programme = np.zeros(norm.shape[1])
        worst_programme = np.zeros(norm.shape[1])
        for t_i in range(norm.shape[1]):
            best_programme[t_i] = max(weighting_norm[:, t_i])
            worst_programme[t_i] = min(weighting_norm[:, t_i])
        g_b = np.zeros(norm.shape[0])
        g_w = np.zeros(norm.shape[0])
        for g_i in range(norm.shape[0]):
            g_b[g_i] = np.sqrt(sum(np.square(weighting_norm[g_i, :] - best_programme)))
            g_w[g_i] = np.sqrt(sum(np.square(weighting_norm[g_i, :] - worst_programme)))
        g = np.zeros(norm.shape[0])
        for k in range(norm.shape[0]):
            g[k] = g_w[k] / (g_w[k] + g_b[k])
        return g

    def write_results_to_file(self,g):
        save_file_path = config.DO_save_file
        data = {'g':list(g)}
        with open(save_file_path, "w") as write_file:
            json.dump(data, write_file)

    def call(self):
        norm=self.normalization()
        w=self.weights(norm)
        g=self.topsis(norm,w)
        # 写回 config
        self.write_results_to_file(g)
        return g

