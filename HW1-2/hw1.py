import cvxpy as cvx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
plt.style.use('ggplot')
# ========================================================
def main():
	# importing data and creating a dataset without outliers
	data = np.genfromtxt("winequality-white.csv", delimiter=";", skip_header=1)
	data_out = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
	data = np.array(data)

	# defining our datasets
	X = data[:,0:-1]
	X_out = data_out[:,0:-1]
	y = data[:, -1]
	y_out = data_out[:, -1]

	# defining our variables for cvxpy
	n = X.shape[1]
	B1 = cvx.Variable(n)
	B2 = cvx.Variable(n)
	B3 = cvx.Variable(n)
	B4 = cvx.Variable(n)

	# defining the cost functions depending on the norm and dataset
	obj1 = cvx.Minimize(cvx.norm(X@B1 - y, 1))
	obj2 = cvx.Minimize(cvx.norm(X @ B2 - y, 2))
	obj1_out = cvx.Minimize(cvx.norm(X_out@B3 - y_out, 1))
	obj2_out = cvx.Minimize(cvx.norm(X_out@B4 - y_out, 2))

	# setting the problems in cvxpy
	prob1 = cvx.Problem(obj1)
	prob2 = cvx.Problem(obj2)
	prob1_out = cvx.Problem(obj1_out)
	prob2_out = cvx.Problem(obj2_out)

	# solving the problems
	prob1.solve()
	prob2.solve()
	prob1_out.solve()
	prob2_out.solve()

	# printing the estimators for each problem
	print("Optimal value for L-1:  ", B1.value)
	print("Optimal value for L-1 (with outliers removed):  ", B3.value)
	print("Optimal value for L-2:  ", B2.value)
	print("Optimal value for L-2 (with outliers removed):  ", B4.value)

	# relative change (to determine robustness to outliers)
	rel_change1 = (B3.value - B1.value) / B1.value #see how much L1 changes
	rel_change2 = (B4.value - B2.value) / B2.value #see how much L2 changes
	print('L1 difference from outliers:', rel_change1, '\nL2 difference from outliers:', rel_change2)
	print('Normed value of L1 and L2 difference vectors:', 'L1:', np.linalg.norm(rel_change1), 'L2:', np.linalg.norm(rel_change2))

	# visualizing the raw dataset and its outliers per column (feature)
	pd.DataFrame(data).plot(kind='box')
	plt.title('Raw Dataset')
	plt.xlabel('Feature')

	# Plotting predictions
	plt.figure()
	plt.title('L1 Robustness Test')
	plt.hist([X @ B1.value, X @ B3.value], bins=10, ec='white', color=['orange', 'teal'], label=['L1 Norm','L1 Norm (without outliers)'], stacked=True)
	plt.legend()
	# plt.hist(X @ B3.value, bins = 10, ec='white', color='teal', label='L1 Norm (without outliers)', stacked=True)

	plt.figure()
	plt.title('L2 Robustness Test')
	plt.hist([X@B2.value, X @ B4.value], bins = 10, ec='white', color=['red', 'blue'],label=['L2 Norm','L2 Norm (without outliers)'], stacked=True)
	plt.legend()
	# plt.hist(X @ B4.value, bins = 10, ec='white', color='blue', label='L2 Norm (without outliers)', stacked=True)

	plt.show()

# ========================================================
if __name__ == "__main__":
	main()