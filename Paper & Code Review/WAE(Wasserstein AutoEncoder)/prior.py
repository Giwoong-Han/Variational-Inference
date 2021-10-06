import numpy as np

def sample_pz(Pz, n_z, batch_size):
    if Pz == 'normal' :
        mean = np.zeros(n_z)
        cov = np.identity(n_z)
        noise = np.random.multivariate_normal(mean, cov, batch_size).astype(np.float32)

        return noise

    elif Pz == 'sphere':
        mean = np.zeros(n_z)
        cov = np.identity(n_z)
        noise = np.random.multivariate_normal(mean, cov, batch_size).astype(np.float32)
        noise = noise / np.sqrt(np.sum(noise * noise, axis=1))[:, np.newaxis]

        return noise

    else : 
        assert f'{Pz} is wrong'