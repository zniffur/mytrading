import pandas as pd
import scipy.optimize as sco
import numpy as np
import math


def calcPortfolioPerf(weights, meanReturns, covMatrix, periods):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''
    # Calculate return and variance

    portReturn = np.sum(meanReturns*weights*periods)
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix*periods, weights)))

    return portReturn, portStdDev


def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate, periods):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix, periods)

    return -(p_ret - riskFreeRate) / p_var


def getPortfolioVol(weights, meanReturns, covMatrix, periods):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix, periods)[1]


def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate, periods):
    '''
    Finds the portfolio of assets providing the maximum Sharpe Ratio

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate, periods)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets, ], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts


def findMinVariancePortfolio(meanReturns, covMatrix, periods):
    '''
    Finds the portfolio of assets providing the lowest volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, periods)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(numAssets))

    opts = sco.minimize(getPortfolioVol, numAssets*[1./numAssets, ], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts


def findEfficientReturn(meanReturns, covMatrix, targetReturn, periods):
    '''
    Finds the portfolio of assets providing the target return with lowest
    volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, periods)

    def getPortfolioReturn(weights):
        return calcPortfolioPerf(weights, meanReturns, covMatrix, periods)[0]

    constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(numAssets))

    return sco.minimize(getPortfolioVol, numAssets*[1./numAssets, ], args=args, method='SLSQP', bounds=bounds, constraints=constraints)


def findEfficientFrontier(meanReturns, covMatrix, rangeOfReturns, periods):
    '''
    Finds the set of portfolios comprising the efficient frontier

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    efficientPortfolios = []
    for ret in rangeOfReturns:
        efficientPortfolios.append(findEfficientReturn(
            meanReturns, covMatrix, ret, periods))

    return efficientPortfolios


def do_mc_simulation(data, assets, numPortfolios, periods, riskFreeRate):

    numAssets = len(assets)
    # *DAILY* returns
    # LOG
    x = data/data.shift(1)
    x.dropna(inplace=True)
    returns = np.log(x)

    # PCT
    #returns = norm_data.pct_change()
    # returns.dropna(inplace=True)

    meanDailyReturns = returns[assets].mean()
    covMatrix = returns[assets].cov()
    # meanDailyReturns

    # Run MC simulation of numPortfolios portfolios

    results = np.zeros((3, numPortfolios))

    # Calculate portfolios
    results = pd.DataFrame(columns=['ret', 'vol', 'sharpe']+assets)

    for i in range(numPortfolios):
        # Draw numAssets random numbers and normalize them to be the portfolio weights

        weights = np.random.random(numAssets)
        weights /= np.sum(weights)

        # Calculate expected return and volatility of portfolio

        pret, pvar = calcPortfolioPerf(
            weights, meanDailyReturns, covMatrix, periods)

        # Convert results to annual basis, calculate Sharpe Ratio, and store them

        # results[0,i] = pret
        # results[1,i] = pvar
        # results[2,i] = (results[0,i] - riskFreeRate)/results[1,i]

        results.loc[i, 'ret'] = pret
        results.loc[i, 'vola'] = pvar
        results.loc[i, 'sharpe'] = (pret - riskFreeRate)/pvar
        for j in range(0, len(weights)):
            results.iloc[i, j+3] = weights[j]

    return results, meanDailyReturns, covMatrix


def do_mc_randomwalk(pct_ret, num_mc_runs, num_years, riskFreeRate, periods):

    u = pct_ret.maxSharpe.mean()
    sigma = pct_ret.maxSharpe.std()

    # Annualized returns
    # MODO 1
    #mu = (norm_data.maxSharpe.iloc[-1]/norm_data.maxSharpe.iloc[0]) ** (252/norm_data.shape[0]) - 1
    #vol = sigma * math.sqrt(252)
    #print(mu, vol)

    # MODO 2
    D = len(pct_ret.maxSharpe)
    mu = pct_ret.maxSharpe.add(1).prod() ** (252 / D) - 1
    vol = sigma * math.sqrt(252)
    #print(mu, vol)

    # MONTECARLO random walk
    mc_runs = pd.DataFrame()

    for i in range(num_mc_runs):
        # create list of daily returns using random normal distribution
        daily_returns = np.random.normal(
            (mu/periods), vol/math.sqrt(periods), periods * num_years)

        random_walk = [1.0]
        for x in daily_returns:
            random_walk.append(random_walk[-1] * (1+x))

        mc_runs[i] = random_walk

    mc_runs_returns = mc_runs.pct_change()
    mc_runs_returns.dropna(inplace=True)

    D = len(mc_runs_returns)
    ann_mc_returns = mc_runs_returns.add(1).prod() ** (periods / D) - 1
    vol_mc_returns = mc_runs_returns.std() * math.sqrt(periods)
    sharpe_mc_runs = (ann_mc_returns - riskFreeRate) / vol_mc_returns

    return mc_runs, ann_mc_returns, vol_mc_returns, sharpe_mc_runs
