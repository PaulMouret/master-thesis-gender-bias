import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, pearsonr, kstest, mannwhitneyu, kruskal, t, normaltest
import numpy as np

import sys
sys.path.insert(1, '/lnet/aic/personal/mouretp/master_thesis2')
#This insertion is ugly and should not be necessary, but for some reason it is on cluster

from utils.global_utils import load_obj_from_jsonfile


std_data = load_obj_from_jsonfile('../saved_objects/noising_experiment/sigma_experiments/', "std_data")

#We xheck data has the expected form
print(f"nb of rows : {len(std_data)}")
print(f"length of a row : {len(std_data[0])}")

#First we print the distribution for one token
first_token = np.array(std_data[0])
print(f"first_token.shape : {first_token.shape}")

plt.hist(first_token, bins=100, range=(-0.2, 0.2), density=True) #bins, range arguments if necessary
plt.xlabel("Value of embedding components")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/hist_first_token.png", dpi=200)
plt.close()

#Then we plot the distribution across all tokens
flat_std_data = np.array(std_data).flatten()
print(f"flat_std_data.shape : {flat_std_data.shape}")

plt.hist(flat_std_data, bins=100, range=(-0.2, 0.2), density=True) #bins, range arguments if necessary
plt.xlabel("Value of embedding components")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/hist_all_tokens.png", dpi=200)
plt.close()

print(f"global mean : {flat_std_data.mean()}")
print(f"global std : {flat_std_data.std()}")

#The normalized equivalent
normalized_flat_std_data = (flat_std_data - flat_std_data.mean()) / (flat_std_data.std())
print(f"normalized_flat_std_data.shape : {normalized_flat_std_data.shape}")

plt.hist(normalized_flat_std_data, bins=100, range=(-0.2, 0.2), density=True) #bins, range arguments if necessary
plt.xlabel("Value of embedding components")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/hist_normalized_all_tokens.png", dpi=200)
plt.close()


#Kolmogorov-Smirnov test for normal distribution

# a. On first token data (this is more an example than anything else)
first_token_test = kstest(rvs=(first_token - first_token.mean()) / (first_token.std()),
                          cdf="norm", alternative="two-sided")[1]
print(f"p-value of Kolmogorov-Smirnov test for first token data : {first_token_test}")

# b. On single token data
#It looks like all token embeddings follow a normal distribution, but a different one; let's see at the same time
ALPHA_VALUE = 0.05
list_pvalues = []
list_means = []
list_stds = []
for i in range(len(std_data)):
    single_token_data = np.array(std_data[i])
    m = single_token_data.mean()
    s = single_token_data.std()
    single_token_test = kstest(rvs=(single_token_data - m) / (s),
                          cdf="norm", alternative="two-sided")[1]
    list_pvalues.append(single_token_test)
    list_means.append(m)
    list_stds.append(s)

nb_normal = sum(1 for pvalue in list_pvalues if pvalue >= ALPHA_VALUE)
print(f"With significance level {ALPHA_VALUE}, {nb_normal} ({nb_normal/len(list_pvalues)}) of tokens have an embedding"
      f" that follows a normal distribution.")

#Now we plot the distribution of means and stds
plt.hist(list_means, bins=100, density=True) #bins, range arguments if necessary
plt.xlabel("Mean value of components for a given embedding")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/hist_mean_embedding.png", dpi=200)
plt.close()

plt.hist(list_stds, bins=100, density=True) #bins, range arguments if necessary
plt.xlabel("Standard deviation of components for a given embedding")
plt.ylabel("Density")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/hist_std_embedding.png", dpi=200)
plt.close()

plt.plot(list_means, list_stds)
plt.xlabel("Mean value of components for a given embedding")
plt.ylabel("Standard deviation of components for a given embedding")
plt.savefig(f"../saved_objects/noising_experiment/sigma_experiments/std_of_mean.png", dpi=200)
plt.close()

# b. On all data
all_data_normal_test = kstest(rvs=normalized_flat_std_data,
                              cdf="norm", alternative="two-sided")[1]
print(f"p-value of Kolmogorov-Smirnov (normal) test for all data : {all_data_normal_test}")

all_data_laplace_test = kstest(rvs=normalized_flat_std_data,
                               cdf="laplace", alternative="two-sided")[1]
print(f"p-value of Kolmogorov-Smirnov (laplace) test for all data : {all_data_laplace_test}")

all_data_logistic_test = kstest(rvs=normalized_flat_std_data,
                                cdf="logistic", alternative="two-sided")[1]
print(f"p-value of Kolmogorov-Smirnov (logistic) test for all data : {all_data_logistic_test}")
