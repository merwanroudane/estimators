import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Set page configuration
st.set_page_config(
    page_title="Econometrics Learning App",
    page_icon="📊",
    layout="wide"
)

# App title and introduction
st.title("🔍 Econometrics Learning App: Bias, Efficiency & Consistency")
st.markdown("""
This app helps you understand three fundamental concepts in econometrics:
- **Bias**: When an estimator's expected value differs from the true parameter value
- **Efficiency**: How precise or variable an estimator is around its expected value
- **Consistency**: How an estimator converges to the true parameter value as sample size increases
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a concept to explore:",
    ["Introduction", "Bias", "Efficiency", "Consistency", "All Concepts Together"]
)

# Introduction page
if page == "Introduction":
    st.header("Introduction to Econometric Estimators")

    st.markdown("""
    ### Welcome to Econometrics!

    Econometrics is about estimating relationships between variables using data and statistical methods.

    ### The Key Properties We'll Explore:

    1. **Bias**: An estimator is unbiased if its expected value equals the true parameter value.

    2. **Efficiency**: An efficient estimator has the smallest variance among all unbiased estimators.

    3. **Consistency**: A consistent estimator converges to the true parameter value as the sample size increases.

    Use the sidebar to navigate between concepts and explore interactive demonstrations!
    """)


# Bias demonstration
elif page == "Bias":
    st.header("Understanding Bias in Estimators")

    st.markdown("""
    Bias occurs when an estimator's expected value differs from the true parameter value.

    In this simulation, we'll generate data from a linear model:

    $Y = \\beta_0 + \\beta_1 X + \\varepsilon$

    And compare two estimators:
    1. **OLS (Ordinary Least Squares)**: An unbiased estimator
    2. **Biased Estimator**: Deliberately biased for demonstration
    """)

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        true_beta0 = st.slider("True Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
        true_beta1 = st.slider("True Slope (β₁)", -10.0, 10.0, 3.0, 0.1)
        sample_size = st.slider("Sample Size", 10, 500, 100)

    with col2:
        noise_level = st.slider("Noise Level (σ)", 0.1, 10.0, 2.0, 0.1)
        n_simulations = st.slider("Number of Simulations", 10, 1000, 200)
        bias_factor = st.slider("Bias Factor", 0.0, 2.0, 0.7, 0.05,
                                help="This introduces bias by shrinking the slope estimate")

    if st.button("Run Bias Simulation"):
        # Create two columns for plots
        col1, col2 = st.columns(2)

        # Simulations for parameter estimates
        beta1_ols = []
        beta1_biased = []

        # Example dataset visualization
        x = np.linspace(-5, 5, sample_size)
        np.random.seed(42)  # For reproducibility
        epsilon = np.random.normal(0, noise_level, sample_size)
        y = true_beta0 + true_beta1 * x + epsilon

        # Run multiple simulations
        for _ in range(n_simulations):
            # Generate new data for each simulation
            x_sim = np.random.uniform(-5, 5, sample_size)
            epsilon_sim = np.random.normal(0, noise_level, sample_size)
            y_sim = true_beta0 + true_beta1 * x_sim + epsilon_sim

            # OLS estimator
            X_sim = sm.add_constant(x_sim)
            model = OLS(y_sim, X_sim).fit()
            beta1_ols.append(model.params[1])

            # Biased estimator (deliberately shrinking coefficient)
            beta1_biased.append(model.params[1] * bias_factor)

        # Plot the distribution of estimates
        with col1:
            st.subheader("Distribution of Estimators")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(beta1_ols, kde=True, color='blue', alpha=0.5, label='OLS Estimator')
            sns.histplot(beta1_biased, kde=True, color='red', alpha=0.5, label='Biased Estimator')
            plt.axvline(true_beta1, color='green', linestyle='dashed', linewidth=2, label='True Value')
            plt.axvline(np.mean(beta1_ols), color='blue', linestyle='dotted', linewidth=2, label='OLS Mean')
            plt.axvline(np.mean(beta1_biased), color='red', linestyle='dotted', linewidth=2, label='Biased Mean')
            plt.legend()
            plt.xlabel('Estimated Slope (β₁)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Estimators over Many Samples')
            st.pyplot(fig)

            # Display bias metrics
            ols_mean = np.mean(beta1_ols)
            biased_mean = np.mean(beta1_biased)
            ols_bias = ols_mean - true_beta1
            biased_bias = biased_mean - true_beta1

            bias_data = pd.DataFrame({
                'Estimator': ['OLS', 'Biased'],
                'Mean Estimate': [ols_mean, biased_mean],
                'True Value': [true_beta1, true_beta1],
                'Bias': [ols_bias, biased_bias],
                'Percent Bias': [100 * ols_bias / true_beta1, 100 * biased_bias / true_beta1]
            })

            st.dataframe(bias_data.round(4))

        # Sample dataset with regression lines
        with col2:
            st.subheader("Example Dataset with Fitted Models")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot data points
            plt.scatter(x, y, alpha=0.5, label='Data Points')

            # True relationship
            plt.plot(x, true_beta0 + true_beta1 * x, 'g-', linewidth=2, label='True Relationship')

            # OLS estimate
            X = sm.add_constant(x)
            model = OLS(y, X).fit()
            plt.plot(x, model.predict(X), 'b--', linewidth=2, label='OLS Estimate')

            # Biased estimate
            biased_slope = model.params[1] * bias_factor
            plt.plot(x, model.params[0] + biased_slope * x, 'r--', linewidth=2, label='Biased Estimate')

            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Sample Dataset with True and Estimated Relationships')
            st.pyplot(fig)

        st.markdown("""
        ### Key Observations:

        - **OLS Estimator**: Centers around the true parameter value (unbiased)
        - **Biased Estimator**: Centers away from the true value (biased)

        The bias of an estimator is the difference between its expected value and the true parameter value.
        """)

# Efficiency demonstration
elif page == "Efficiency":
    st.header("Understanding Efficiency in Estimators")

    st.markdown("""
    Efficiency refers to how precise an estimator is. An efficient estimator has the smallest variance among all unbiased estimators.

    In this simulation, we'll compare two unbiased estimators with different variances:
    1. **OLS (Ordinary Least Squares)**: Efficient under certain conditions
    2. **Inefficient Estimator**: Has larger variance
    """)

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        true_beta0 = st.slider("True Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
        true_beta1 = st.slider("True Slope (β₁)", -10.0, 10.0, 3.0, 0.1)
        sample_size = st.slider("Sample Size", 10, 500, 100)

    with col2:
        noise_level = st.slider("Noise Level (σ)", 0.1, 10.0, 2.0, 0.1)
        n_simulations = st.slider("Number of Simulations", 10, 1000, 200)
        inefficiency_factor = st.slider("Inefficiency Factor", 1.0, 5.0, 1.5, 0.1,
                                        help="This introduces additional variance to the inefficient estimator")

    if st.button("Run Efficiency Simulation"):
        # Create two columns for plots
        col1, col2 = st.columns(2)

        # Simulations for parameter estimates
        beta1_ols = []
        beta1_inefficient = []

        # Example dataset visualization
        x = np.linspace(-5, 5, sample_size)
        np.random.seed(42)  # For reproducibility
        epsilon = np.random.normal(0, noise_level, sample_size)
        y = true_beta0 + true_beta1 * x + epsilon

        # Run multiple simulations
        for i in range(n_simulations):
            # Generate new data for each simulation
            x_sim = np.random.uniform(-5, 5, sample_size)
            epsilon_sim = np.random.normal(0, noise_level, sample_size)
            y_sim = true_beta0 + true_beta1 * x_sim + epsilon_sim

            # OLS estimator
            X_sim = sm.add_constant(x_sim)
            model = OLS(y_sim, X_sim).fit()
            beta1_ols.append(model.params[1])

            # Inefficient estimator (add random noise to OLS estimator)
            # This keeps it unbiased but increases variance
            additional_noise = np.random.normal(0, model.bse[1] * (inefficiency_factor - 1))
            beta1_inefficient.append(model.params[1] + additional_noise)

        # Plot the distribution of estimates
        with col1:
            st.subheader("Distribution of Estimators")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(beta1_ols, kde=True, color='blue', alpha=0.5, label='OLS (Efficient)')
            sns.histplot(beta1_inefficient, kde=True, color='orange', alpha=0.5, label='Inefficient Estimator')
            plt.axvline(true_beta1, color='green', linestyle='dashed', linewidth=2, label='True Value')
            plt.axvline(np.mean(beta1_ols), color='blue', linestyle='dotted', linewidth=2, label='OLS Mean')
            plt.axvline(np.mean(beta1_inefficient), color='orange', linestyle='dotted', linewidth=2,
                        label='Inefficient Mean')
            plt.legend()
            plt.xlabel('Estimated Slope (β₁)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Estimators over Many Samples')
            st.pyplot(fig)

            # Display efficiency metrics
            ols_mean = np.mean(beta1_ols)
            inefficient_mean = np.mean(beta1_inefficient)
            ols_var = np.var(beta1_ols)
            inefficient_var = np.var(beta1_inefficient)

            efficiency_data = pd.DataFrame({
                'Estimator': ['OLS (Efficient)', 'Inefficient'],
                'Mean Estimate': [ols_mean, inefficient_mean],
                'True Value': [true_beta1, true_beta1],
                'Variance': [ols_var, inefficient_var],
                'Standard Error': [np.sqrt(ols_var), np.sqrt(inefficient_var)],
                'Relative Efficiency': [1.0, ols_var / inefficient_var]
            })

            st.dataframe(efficiency_data.round(4))

        # Confidence intervals visualization
        with col2:
            st.subheader("Confidence Intervals Comparison")

            # Create synthetic estimates for visualization
            np.random.seed(123)
            n_estimates = 20
            ols_estimates = np.random.normal(true_beta1, np.sqrt(ols_var), n_estimates)
            inefficient_estimates = np.random.normal(true_beta1, np.sqrt(inefficient_var), n_estimates)

            # Confidence intervals
            ols_ci_low = ols_estimates - 1.96 * np.sqrt(ols_var)
            ols_ci_high = ols_estimates + 1.96 * np.sqrt(ols_var)
            inefficient_ci_low = inefficient_estimates - 1.96 * np.sqrt(inefficient_var)
            inefficient_ci_high = inefficient_estimates + 1.96 * np.sqrt(inefficient_var)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot OLS confidence intervals
            for i in range(n_estimates):
                plt.plot([0, 1], [ols_estimates[i], ols_estimates[i]], 'bo-', alpha=0.3)
                plt.plot([0, 0], [ols_ci_low[i], ols_ci_high[i]], 'b-', alpha=0.3)
                plt.plot([1, 1], [inefficient_ci_low[i], inefficient_ci_high[i]], 'orange', alpha=0.3)

            # Add the mean estimates with wider lines
            plt.plot([0, 1], [np.mean(ols_estimates), np.mean(inefficient_estimates)], 'k--', linewidth=2)

            # Add horizontal line for true value
            plt.axhline(true_beta1, color='green', linestyle='dashed', linewidth=2, label='True Value')

            # Customize the plot
            plt.xlim(-0.5, 1.5)
            plt.xticks([0, 1], ['OLS (Efficient)', 'Inefficient'])
            plt.ylabel('Estimated Slope (β₁)')
            plt.title('95% Confidence Intervals Comparison')
            st.pyplot(fig)

        st.markdown("""
        ### Key Observations:

        - Both estimators are unbiased (centered around the true value)
        - The **OLS Estimator** has smaller variance (more precise)
        - The **Inefficient Estimator** has larger variance (less precise)

        An efficient estimator gives more precise estimates, resulting in narrower confidence intervals and more reliable inference.
        """)

# Consistency demonstration
elif page == "Consistency":
    st.header("Understanding Consistency in Estimators")

    st.markdown("""
    Consistency means that as the sample size increases, the estimator converges to the true parameter value.

    In this simulation, we'll examine how estimators behave with increasing sample sizes:
    1. **OLS (Ordinary Least Squares)**: A consistent estimator
    2. **Inconsistent Estimator**: Doesn't converge to the true value as sample size increases
    """)

    # Parameters
    col1, col2 = st.columns(2)

    with col1:
        true_beta0 = st.slider("True Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
        true_beta1 = st.slider("True Slope (β₁)", -10.0, 10.0, 3.0, 0.1)
        max_sample_size = st.slider("Maximum Sample Size", 100, 5000, 1000)

    with col2:
        noise_level = st.slider("Noise Level (σ)", 0.1, 10.0, 2.0, 0.1)
        n_simulations = st.slider("Simulations per Sample Size", 5, 100, 20)
        inconsistency_level = st.slider("Inconsistency Level", 0.0, 1.0, 0.5, 0.05,
                                        help="Higher values make the inconsistent estimator more severely inconsistent")

    if st.button("Run Consistency Simulation"):
        # Define sample sizes to test
        sample_sizes = np.geomspace(20, max_sample_size, 10).astype(int)

        # Store results
        results = []

        # Progress bar
        progress_bar = st.progress(0)

        # Run simulations for different sample sizes
        for i, size in enumerate(sample_sizes):
            beta1_ols_samples = []
            beta1_inconsistent_samples = []

            for _ in range(n_simulations):
                # Generate data
                x_sim = np.random.uniform(-5, 5, size)
                epsilon_sim = np.random.normal(0, noise_level, size)
                y_sim = true_beta0 + true_beta1 * x_sim + epsilon_sim

                # OLS estimator
                X_sim = sm.add_constant(x_sim)
                model = OLS(y_sim, X_sim).fit()
                beta1_ols_samples.append(model.params[1])

                # Inconsistent estimator (biased and doesn't converge)
                # This could be a misspecified model that systematically misses a part of the data
                # For example, we'll use only a subset of the data based on sample size
                subsample_size = int(size * (1 - inconsistency_level))
                X_sub = sm.add_constant(x_sim[:subsample_size])
                y_sub = y_sim[:subsample_size]
                if subsample_size > 1:  # Ensure we have enough data for regression
                    model_inconsistent = OLS(y_sub, X_sub).fit()
                    beta1_inconsistent_samples.append(model_inconsistent.params[1])
                else:
                    beta1_inconsistent_samples.append(np.nan)

            # Calculate mean and variance for this sample size
            results.append({
                'sample_size': size,
                'ols_mean': np.mean(beta1_ols_samples),
                'ols_variance': np.var(beta1_ols_samples),
                'inconsistent_mean': np.mean(beta1_inconsistent_samples),
                'inconsistent_variance': np.var(beta1_inconsistent_samples)
            })

            # Update progress bar
            progress_bar.progress((i + 1) / len(sample_sizes))

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        # Create two columns for plots
        col1, col2 = st.columns(2)

        # Plot mean estimates vs sample size
        with col1:
            st.subheader("Convergence of Estimators")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(results_df['sample_size'], results_df['ols_mean'], 'bo-', label='OLS Estimator')
            plt.plot(results_df['sample_size'], results_df['inconsistent_mean'], 'ro-', label='Inconsistent Estimator')
            plt.axhline(true_beta1, color='green', linestyle='dashed', linewidth=2, label='True Value')
            plt.xscale('log')  # Log scale for sample size
            plt.xlabel('Sample Size (log scale)')
            plt.ylabel('Mean Estimated Slope (β₁)')
            plt.title('Convergence of Estimators with Increasing Sample Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Plot variance vs sample size
        with col2:
            st.subheader("Variance of Estimators")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(results_df['sample_size'], results_df['ols_variance'], 'bo-', label='OLS Estimator')
            plt.plot(results_df['sample_size'], results_df['inconsistent_variance'], 'ro-',
                     label='Inconsistent Estimator')
            plt.xscale('log')  # Log scale for sample size
            plt.yscale('log')  # Log scale for variance
            plt.xlabel('Sample Size (log scale)')
            plt.ylabel('Variance of Estimated Slope (log scale)')
            plt.title('Variance of Estimators with Increasing Sample Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

        # Display numeric results
        st.subheader("Numerical Results")
        display_df = results_df.copy()
        display_df['ols_bias'] = display_df['ols_mean'] - true_beta1
        display_df['inconsistent_bias'] = display_df['inconsistent_mean'] - true_beta1
        display_df = display_df.rename(columns={
            'sample_size': 'Sample Size',
            'ols_mean': 'OLS Mean',
            'ols_variance': 'OLS Variance',
            'ols_bias': 'OLS Bias',
            'inconsistent_mean': 'Inconsistent Mean',
            'inconsistent_variance': 'Inconsistent Variance',
            'inconsistent_bias': 'Inconsistent Bias'
        })
        st.dataframe(display_df.round(4))

        st.markdown("""
        ### Key Observations:

        - **OLS Estimator**: Converges to the true parameter value as sample size increases (consistent)
        - **Inconsistent Estimator**: Does not converge to the true value (inconsistent)
        - Variance of both estimators decreases with sample size, but consistency is about the bias component

        A consistent estimator becomes more accurate as you collect more data, which is a crucial property for reliable statistical inference.
        """)

# All concepts together
elif page == "All Concepts Together":
    st.header("Understanding Bias, Efficiency, and Consistency Together")

    st.markdown("""
    Let's explore all three properties together to understand how they interact. We'll compare four types of estimators:

    1. **BLUE**: Best Linear Unbiased Estimator (unbiased, efficient, consistent)
    2. **Biased but Efficient**: Has systematic error but low variance
    3. **Unbiased but Inefficient**: No systematic error but high variance
    4. **Inconsistent**: Doesn't converge to true value as sample size increases
    """)

    # Parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        true_beta0 = st.slider("True Intercept (β₀)", -10.0, 10.0, 2.0, 0.1)
        true_beta1 = st.slider("True Slope (β₁)", -10.0, 10.0, 3.0, 0.1)

    with col2:
        sample_size = st.slider("Sample Size", 10, 1000, 100)
        noise_level = st.slider("Noise Level (σ)", 0.1, 10.0, 2.0, 0.1)

    with col3:
        n_simulations = st.slider("Number of Simulations", 10, 1000, 200)
        bias_factor = st.slider("Bias Factor", 0.0, 1.0, 0.7, 0.05)

    if st.button("Run Comprehensive Simulation"):
        # Create progress bar
        progress_bar = st.progress(0)

        # Storage for simulations
        blue_estimates = []
        biased_efficient_estimates = []
        unbiased_inefficient_estimates = []
        inconsistent_estimates = []

        # Run simulations
        for i in range(n_simulations):
            # Generate data
            x_sim = np.random.uniform(-5, 5, sample_size)
            epsilon_sim = np.random.normal(0, noise_level, sample_size)
            y_sim = true_beta0 + true_beta1 * x_sim + epsilon_sim

            # 1. BLUE estimator (OLS)
            X_sim = sm.add_constant(x_sim)
            model = OLS(y_sim, X_sim).fit()
            blue_estimates.append(model.params[1])

            # 2. Biased but efficient estimator (Ridge-like)
            biased_efficient_estimates.append(model.params[1] * bias_factor)

            # 3. Unbiased but inefficient estimator
            # Simulate by adding noise to the OLS estimate
            inefficiency_noise = np.random.normal(0, 2 * model.bse[1])
            unbiased_inefficient_estimates.append(model.params[1] + inefficiency_noise)

            # 4. Inconsistent estimator
            # Use only a subset of data
            subset_size = max(int(sample_size * 0.5), 2)
            subset_idx = np.random.choice(sample_size, subset_size, replace=False)
            X_sub = sm.add_constant(x_sim[subset_idx])
            y_sub = y_sim[subset_idx]
            model_inconsistent = OLS(y_sub, X_sub).fit()
            inconsistent_estimates.append(model_inconsistent.params[1])

            # Update progress
            progress_bar.progress((i + 1) / n_simulations)

        # Create columns for visualization
        col1, col2 = st.columns(2)

        # Distribution plot
        with col1:
            st.subheader("Distribution of Different Estimators")
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot distributions
            sns.kdeplot(blue_estimates, color='blue', label='BLUE (Best Linear Unbiased Estimator)')
            sns.kdeplot(biased_efficient_estimates, color='red', label='Biased but Efficient')
            sns.kdeplot(unbiased_inefficient_estimates, color='green', label='Unbiased but Inefficient')
            sns.kdeplot(inconsistent_estimates, color='purple', label='Inconsistent')

            # Add true value
            plt.axvline(true_beta1, color='black', linestyle='dashed', linewidth=2, label='True Value')

            # Add mean values
            plt.axvline(np.mean(blue_estimates), color='blue', linestyle='dotted', linewidth=1)
            plt.axvline(np.mean(biased_efficient_estimates), color='red', linestyle='dotted', linewidth=1)
            plt.axvline(np.mean(unbiased_inefficient_estimates), color='green', linestyle='dotted', linewidth=1)
            plt.axvline(np.mean(inconsistent_estimates), color='purple', linestyle='dotted', linewidth=1)

            plt.legend()
            plt.xlabel('Estimated Slope (β₁)')
            plt.ylabel('Density')
            plt.title('Distribution of Different Estimators')
            st.pyplot(fig)

        # Summary statistics
        with col2:
            st.subheader("Properties of Estimators")

            # Calculate metrics
            estimator_props = pd.DataFrame({
                'Estimator': ['BLUE', 'Biased Efficient', 'Unbiased Inefficient', 'Inconsistent'],
                'Mean': [
                    np.mean(blue_estimates),
                    np.mean(biased_efficient_estimates),
                    np.mean(unbiased_inefficient_estimates),
                    np.mean(inconsistent_estimates)
                ],
                'Bias': [
                    np.mean(blue_estimates) - true_beta1,
                    np.mean(biased_efficient_estimates) - true_beta1,
                    np.mean(unbiased_inefficient_estimates) - true_beta1,
                    np.mean(inconsistent_estimates) - true_beta1
                ],
                'Variance': [
                    np.var(blue_estimates),
                    np.var(biased_efficient_estimates),
                    np.var(unbiased_inefficient_estimates),
                    np.var(inconsistent_estimates)
                ],
                'MSE': [
                    np.mean((np.array(blue_estimates) - true_beta1) ** 2),
                    np.mean((np.array(biased_efficient_estimates) - true_beta1) ** 2),
                    np.mean((np.array(unbiased_inefficient_estimates) - true_beta1) ** 2),
                    np.mean((np.array(inconsistent_estimates) - true_beta1) ** 2)
                ]
            })

            # Add bias-variance decomposition
            estimator_props['Bias²'] = estimator_props['Bias'] ** 2
            estimator_props['Bias² + Variance'] = estimator_props['Bias²'] + estimator_props['Variance']

            # Display properties table
            st.dataframe(estimator_props.round(4))

            # Create visualization of bias-variance tradeoff
            fig, ax = plt.subplots(figsize=(10, 6))

            # Data for plotting
            estimators = estimator_props['Estimator']
            bias_squared = estimator_props['Bias²']
            variance = estimator_props['Variance']

            # Create stacked bar chart
            ax.bar(estimators, bias_squared, label='Bias²', color='skyblue')
            ax.bar(estimators, variance, bottom=bias_squared, label='Variance', color='lightcoral')

            ax.set_ylabel('Error Components')
            ax.set_title('Bias-Variance Decomposition of MSE')
            ax.legend()

            st.pyplot(fig)

        # Summary of concepts
        st.subheader("Summary of Estimator Properties")

        st.markdown("""
        ### Bias, Efficiency, and Consistency Together:

        1. **BLUE Estimator (OLS)**: 
           - Unbiased: Expected value equals the true parameter
           - Efficient: Lowest variance among unbiased estimators
           - Consistent: Converges to true value as sample size increases

        2. **Biased but Efficient**:
           - Biased: Expected value differs from true parameter
           - Very efficient: Low variance (often lower than BLUE)
           - May have lower Mean Squared Error (MSE) than unbiased estimators

        3. **Unbiased but Inefficient**:
           - Unbiased: Expected value equals the true parameter
           - Inefficient: Has high variance
           - May have worse practical performance despite being unbiased

        4. **Inconsistent**:
           - Does not converge to true value as sample size increases
           - Generally the least desirable property
           - Having more data doesn't guarantee better estimates

        ### The Bias-Variance Tradeoff

        Mean Squared Error (MSE) = Bias² + Variance

        Sometimes accepting a little bias can substantially reduce variance, leading to a lower overall MSE and better predictions.
        """)

        # Visual summary
        st.subheader("Visual Summary of Properties")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Bias vs Variance
        ax1.scatter(
            estimator_props['Bias'],
            estimator_props['Variance'],
            s=100, alpha=0.7
        )

        # Add labels for each point
        for i, txt in enumerate(estimator_props['Estimator']):
            ax1.annotate(txt,
                         (estimator_props['Bias'][i], estimator_props['Variance'][i]),
                         xytext=(5, 5), textcoords='offset points')

        ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Bias')
        ax1.set_ylabel('Variance')
        ax1.set_title('Bias vs. Variance')
        ax1.grid(True, alpha=0.3)

        # Plot 2: MSE Comparison
        ax2.bar(estimator_props['Estimator'], estimator_props['MSE'], alpha=0.7)
        ax2.set_ylabel('Mean Squared Error (MSE)')
        ax2.set_title('Total Error Comparison')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

# Add a note at the bottom of all pages
st.markdown("---")
st.caption("Created by the Dr Merwan Roudane. Designed for educational purposes.")