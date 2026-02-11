import numpy as np


def create_stratified_splits(samples, n_bins=8):
    """
    Create temporally stratified splits by binning samples by age and sampling from each bin.
    Returns a sampling function that maintains temporal distribution.
    """
    # Sort samples by age
    samples_by_age = sorted(enumerate(samples), key=lambda x: x[1]['age'])
    
    # Create bins of equal size
    n_samples = len(samples_by_age)
    bin_size = n_samples // n_bins
    bins = []
    
    # Distribute samples into bins
    for i in range(0, n_samples, bin_size):
        if i + bin_size > n_samples and bins:
            # Add remaining samples to the last bin
            bins[-1].extend(samples_by_age[i:])
        else:
            bins.append(samples_by_age[i:i + bin_size])

    def sample_from_bins(samples, sample_fraction):
        selected_indices = []
        for bin_samples in bins:
            n_to_sample = max(1, int(len(bin_samples) * sample_fraction))
            # Randomly sample from this bin
            selected = np.random.choice(len(bin_samples), n_to_sample, replace=False)
            # Each bin_sample is a tuple of (index, sample)
            selected_indices.extend([bin_samples[idx][0] for idx in selected])
        return selected_indices

    return sample_from_bins
