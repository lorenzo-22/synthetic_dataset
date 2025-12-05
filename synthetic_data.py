#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse

def generate_data():
    """Simulates proteomics data with a signal in the first 200 proteins."""
    print("Initializing simulation...")
    
    # 1. Setup parameters
    np.random.seed(42)
    proteins_n = 2000
    nA, nB = 6, 6
    mu, sigma = 20, 0.5
    
    # 2. Generate random normal distribution data
    A = np.random.normal(mu, sigma, size=(proteins_n, nA))
    B = np.random.normal(mu, sigma, size=(proteins_n, nB))
    
    # 3. Inject signal (+1 log2 effect) into first 200 proteins of Group B
    B[:200] += 1.0
    
    # 4. Create label tracker (optional, but good for validation)
    true_labels = np.zeros(proteins_n, dtype=int)
    true_labels[:200] = 1

    # 5) Instrument noise (Gaussian, proportional to protein variability)
    # Mass spectrometers introduce random fluctuations in measured intensities. The noise is 15% of each proteins variability (adds jitter to values).
    noise_A = np.random.normal(loc=0.0, scale=0.15 * A.std(axis=1, keepdims=True), size=A.shape)
    noise_B = np.random.normal(loc=0.0, scale=0.15 * B.std(axis=1, keepdims=True), size=B.shape)
    A += noise_A
    B += noise_B
    
    # 2) Missing values (missing not at random=bias toward low-abundance; plus some missing at random=due to sampling)
    # Compute global 25th percentile to bias toward low values
    q25_A = np.nanpercentile(A, 25)
    q25_B = np.nanpercentile(B, 25)
    
    # Missing not at random: more missing when below 25th percentile
    mnar_A = (A < q25_A) & (np.random.rand(*A.shape) < 0.25)  # ~25% dropout low-abundance
    mnar_B = (B < q25_B) & (np.random.rand(*B.shape) < 0.25)
    
    # Missing at random: random missing everywhere (~5%)
    mar_A = np.random.rand(*A.shape) < 0.05
    mar_B = np.random.rand(*B.shape) < 0.05
    
    # Set these to NAN
    A[mnar_A | mar_A] = np.nan
    B[mnar_B | mar_B] = np.nan
    
    # 3) Batch effects (systematic shift in half of group B samples)
    # Here, we simulate an instrument drift as +0.5 in B samples 3 to5
    batch_labels = np.array(["A"] * nA + ["B"] * nB)
    batch_shift_B = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    B += batch_shift_B 
    
    # 6. formatting into DataFrame
    counts = np.hstack([A, B])
    samples = [f"A{i}" for i in range(nA)] + [f"B{i}" for i in range(nB)]
    proteins = [f"Prot{i}" for i in range(proteins_n)]
    
    df = pd.DataFrame(counts, index=proteins, columns=samples)
    
    imputed_df = df.copy()
    for col in imputed_df.columns:
	col_values = imputed_df[col].values
	if np.isnan(col_values).any():
		low_q = np.nanpercentile(col_values, 1)
		# add small jitter to avoid identical values
		jitter = np.random.normal(loc=0.0, scale=0.05, size=col_values.shape)
		col_values[np.isnan(col_values)] = low_q + jitter[np.isnan(col_values)]
		imputed_df[col] = col_values
    df = imputed_df.copy()

    # Add the truth label as a column for reference (optional)
    df['is_differentially_expressed'] = true_labels
    
    return df

def make_sample_labels(df):
    # Make true sample labels (group of the sample, here A and B made into 0s and 1s for standardization)
    sample_labels = {sample: ("1" if sample.startswith('A') else '0') for sample in df.T.index}
    sample_labels = df = pd.DataFrame.from_dict(sample_labels, orient='index', columns=['label'])

    return sample_labels

if __name__ == "__main__":
    # Setup Argument Parser for OmniBenchmark
    parser = argparse.ArgumentParser(description="Generate synthetic proteomics data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--name", type=str, required=True,
                        help="Dataset name")
    
    args = parser.parse_args()

    # Generate the dataframe
    df = generate_data()
    
    # Create output directory if needed
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct output file paths
    output_file = os.path.join(args.output_dir, f"{args.name}.dataset.csv")
    protein_labels_file = os.path.join(args.output_dir, f"{args.name}.true_labels_proteins.csv")
    sample_labels_file = os.path.join(args.output_dir, f"{args.name}.true_labels.csv")
    
    # Print summary to console
    print(f"\nData Generated Successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save main dataset to CSV (without labels column)
    df_data = df.drop(columns=['is_differentially_expressed'])
    df_data.to_csv(output_file, na_rep='NaN')
    print(f"\nSaved dataset to '{output_file}'")
    
    # Save labels separately
    labels_df = df[['is_differentially_expressed']]
    labels_df.to_csv(protein_labels_file, header=False)
    print(f"Saved true labels to '{protein_labels_file}'")

    # Save sample lables separately
    sample_labels = make_sample_labels(df_data)
    sample_labels.to_csv(sample_labels_file, header=False)
    print(f"Saved true sample labels to '{sample_labels_file}'")
