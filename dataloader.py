import numpy as np


class data_loader:

    def __init__(self, type, batch_size=32):
        self.type = type
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return None

class IHDP_loader(data_loader):

    def __init__(self, i, type, in_sample=False, shuffle=True, batch_size=32):
        # i is for repeated simulation
        if type=='train' or type=='valid':
            data = np.load('../data/IHDP/ihdp_npci_1-1000.train.npz')
        elif type=='test':
            if in_sample:
                data = np.load('../data/IHDP/ihdp_npci_1-1000.train.npz')
            else:
                data = np.load('../data/IHDP/ihdp_npci_1-1000.test.npz')
        self.x = data['x'][:,:,i]
        self.t = data['t'][:,i]
        self.yf = data['yf'][:,i]
        self.ycf = data['ycf'][:,i]
        self.mu0 = data['mu0'][:,i]
        self.mu1 = data['mu1'][:,i]
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in xrange(25) if i not in self.binfeats]

        self.train_size = int(0.7 * len(self.x))
        if type=='train':
            self.x = self.x[:self.train_size]
            self.t = self.t[:self.train_size]
            self.yf = self.yf[:self.train_size]
            self.ycf = self.ycf[:self.train_size]
            self.mu0 = self.mu0[:self.train_size]
            self.mu1 = self.mu1[:self.train_size]
        elif type=='valid':
            self.x = self.x[self.train_size:]
            self.t = self.t[self.train_size:]
            self.yf = self.yf[self.train_size:]
            self.ycf = self.ycf[self.train_size:]
            self.mu0 = self.mu0[self.train_size:]
            self.mu1 = self.mu1[self.train_size:]

        self.x_mean = np.mean(self.x, axis=0)
        self.yf_0_mean = np.mean(self.yf[self.t==0], axis=0)
        self.yf_1_mean = np.mean(self.yf[self.t==1], axis=0)

        super().__init__(type=type, batch_size=batch_size)

    def __next__(self):
        # work as generator for time efficiency
        if self.type=='train':
            mini_arr = np.random.choice(list(range(self.train_size)), self.batch_size)
            mini_x = self.x[mini_arr]
            mini_t = self.t[mini_arr]
            mini_yf = self.yf[mini_arr]
            mini_ycf = self.ycf[mini_arr]
            mini_mu0 = self.mu0[mini_arr]
            mini_mu1 = self.mu1[mini_arr]
            return mini_x, mini_t, mini_yf, mini_ycf, mini_mu0, mini_mu1

        else:
            return self.x, self.t, self.yf, self.ycf, self.mu0, self.mu1

class TWINS_loader(data_loader):

    def __init__(self, i, type, in_sample=False, shuffle=True, batch_size=32):
        # i is for repeated simulation
        if type=='train' or type=='valid':
            data = np.load('../data/TWINS/twins_1-10.train.npz')
        elif type=='test':
            if in_sample:
                data = np.load('../data/TWINS/twins_1-10.train.npz')
            else:
                data = np.load('../data/TWINS/twins_1-10.test.npz')
        self.x = data['x'][:,:,i]
        self.t = data['t'][:,i]
        self.yf = data['yf'][:,i]
        self.ycf = data['ycf'][:,i]

        self.train_size = int(0.7 * len(self.x))
        if type=='train':
            self.x = self.x[:self.train_size]
            self.t = self.t[:self.train_size]
            self.yf = self.yf[:self.train_size]
            self.ycf = self.ycf[:self.train_size]
        elif type=='valid':
            self.x = self.x[self.train_size:]
            self.t = self.t[self.train_size:]
            self.yf = self.yf[self.train_size:]
            self.ycf = self.ycf[self.train_size:]

        self.x_mean = np.mean(self.x, axis=0)
        self.yf_0_mean = np.mean(self.yf[self.t==0], axis=0)
        self.yf_1_mean = np.mean(self.yf[self.t==1], axis=0)

        super().__init__(type=type, batch_size=batch_size)

    def __next__(self):
        # work as generator for time efficiency
        if self.type=='train':
            mini_arr = np.random.choice(list(range(self.train_size)), self.batch_size, replace=False)
            mini_x = self.x[mini_arr]
            mini_t = self.t[mini_arr]
            mini_yf = self.yf[mini_arr]
            mini_ycf = self.ycf[mini_arr]
            return mini_x, mini_t, mini_yf, mini_ycf

        else:
            return self.x, self.t, self.yf, self.ycf

class JOBS_loader(data_loader):
    def __init__(self, i, type, in_sample=False, shuffle=True, batch_size=32):
        # i is for repeated simulation
        if type=='train' or type=='valid':
            data = np.load('../data/JOBS/jobs.train.npz')
        elif type=='test':
            if in_sample:
                data = np.load('../data/JOBS/jobs.train.npz')
            else:
                data = np.load('../data/JOBS/jobs.test.npz')
        self.x = data['x'][:,:,i]
        self.t = data['t'][:,i]
        self.e = data['e'][:,i]
        self.yf = data['yf'][:,i]

        self.train_size = int(0.7 * len(self.x))

        if type=='train':
            self.x = self.x[:self.train_size]
            self.t = self.t[:self.train_size]
            self.e = self.e[:self.train_size]
            self.yf = self.yf[:self.train_size]
        elif type=='valid':
            self.x = self.x[self.train_size:]
            self.t = self.t[self.train_size:]
            self.e = self.e[self.train_size:]
            self.yf = self.yf[self.train_size:]

        self.x_mean = np.mean(self.x, axis=0)
        self.yf_0_mean = np.mean(self.yf[self.t==0], axis=0)
        self.yf_1_mean = np.mean(self.yf[self.t==1], axis=0)

        super().__init__(type=type, batch_size=batch_size)

    def __next__(self):
        # work as generator for time efficiency
        if self.type=='train':
            mini_arr = np.random.choice(list(range(self.train_size)), self.batch_size, replace=False)
            mini_x = self.x[mini_arr]
            mini_t = self.t[mini_arr]
            mini_e = self.e[mini_arr]
            mini_yf = self.yf[mini_arr]
            return mini_x, mini_t, mini_e, mini_yf

        else:
            return self.x, self.t, self.e, self.yf
