"""
2D φ-Fixed Toy for CQFT: Wolff Cluster Analysis
Enhanced with cluster size logging and visualization
Focus: Lattices L ∈ [128, 256, 512, 1024, 1536]
PARALLEL: multiprocessing for independent lattice sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import logging
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time
warnings.filterwarnings('ignore')

# Setup logging (thread-safe with process ID)
def setup_logger(L_cur):
    """Create process-specific logger"""
    logger = logging.getLogger(f'WolffCluster_L{L_cur}')
    logger.setLevel(logging.INFO)
    
    # Create handler if not exists
    if not logger.handlers:
        handler = logging.FileHandler(f'wolff_L{L_cur}.log', mode='w')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
    
    return logger

# ========== CONSTANTS ==========
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈1.618
ETA_TARGET = 0.809
LS = [128, 256, 512, 1024, 1536]  # Large lattices only
BETA_C = np.log(1 + PHI) / 2  # ≈ 0.481
G_YUK = 1 / PHI
GAMMA_DEC = 1 / PHI**2
THETA_TWIST = np.pi / PHI

np.random.seed(42)

# ========== POWER LAW ==========
def power_law(r, A, eta_loc):
    """G(r) ~ A / r^eta_loc"""
    return A / r**eta_loc


# ========== LATTICE KERNEL ==========
def phi_kernel(L, sigma=None):
    """φ-weighted interaction kernel"""
    if sigma is None:
        sigma = PHI
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    r = np.sqrt((x - L/2)**2 + (y - L/2)**2)
    r[r == 0] = 1e-6
    kern = 1 / r**PHI * np.exp(-r / sigma)
    return kern / kern.sum()


# ========== METROPOLIS UPDATE ==========
def metropolis_step(spins, beta, kernel, g_yuk, theta_twist):
    """Single Metropolis update with Yukawa noise"""
    L = spins.shape[0]
    
    # FFT convolution for efficiency
    if kernel.shape[0] > 32:
        half = 16
        kernel_trunc = kernel[L//2 - half:L//2 + half,
                              L//2 - half:L//2 + half]
        s_pad = np.pad(spins, ((half, half), (half, half)), mode='wrap')
        energy_field = fftconvolve(s_pad, kernel_trunc, mode='same')[:L, :L]
    else:
        energy_field = convolve(spins, kernel, mode='wrap')
    
    i, j = np.random.randint(0, L, 2)
    spins_new = spins.copy()
    spins_new[i, j] *= -1
    
    dE = -2 * spins[i, j] * energy_field[i, j]
    dE += g_yuk * np.random.randn()
    
    delta_sigma = spins_new[i, j] - spins[i, j]
    if i == 0 or i == L - 1:
        dE += theta_twist * np.sin(2 * np.pi * j / L) * delta_sigma
    
    accept = (dE < 0) or (np.random.rand() < np.exp(-beta * dE))
    if accept:
        spins[i, j] = spins_new[i, j]
    
    return spins, accept


# ========== WOLFF CLUSTER (ENHANCED LOGGING) ==========
def wolff_cluster(spins, beta, logger=None, log_data=True):
    """
    Wolff cluster flip with detailed logging
    Returns: updated spins, cluster_size
    """
    L = spins.shape[0]
    visited = np.zeros_like(spins, dtype=bool)
    flip = np.zeros_like(spins, dtype=bool)
    
    # Random seed
    i, j = np.random.randint(0, L, 2)
    seed_spin = spins[i, j]
    stack = [(i, j)]
    visited[i, j] = True
    
    # Build cluster
    p_add = 1 - np.exp(-2 * beta)
    while stack:
        ci, cj = stack.pop()
        flip[ci, cj] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = (ci + di) % L, (cj + dj) % L
            if not visited[ni, nj] and spins[ni, nj] == seed_spin:
                if np.random.rand() < p_add:
                    visited[ni, nj] = True
                    stack.append((ni, nj))
    
    cluster_size = np.sum(flip)
    cluster_fraction = cluster_size / (L * L)
    
    # Log cluster statistics
    if log_data and logger is not None:
        logger.info(f"L={L}, size={cluster_size}, "
                   f"frac={cluster_fraction:.4f}, "
                   f"p_add={p_add:.4f}, "
                   f"seed=({i},{j})")
    
    spins[flip] *= -1
    return spins, cluster_size


# ========== CORRELATION FUNCTION ==========
def corr_2d(spins, r_max):
    """Compute radial correlation function G(r)"""
    L = spins.shape[0]
    center = L // 2
    corr = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    
    for dx in range(-r_max, r_max + 1):
        for dy in range(-r_max, r_max + 1):
            r = int(np.sqrt(dx**2 + dy**2))
            if 1 <= r <= r_max:
                xi, yi = center + dx, center + dy
                if 0 <= xi < L and 0 <= yi < L:
                    val = spins[center, center] * spins[xi, yi]
                    corr[r] += val
                    counts[r] += 1
    
    corr /= np.maximum(counts, 1)
    
    # Decoherence factor (reduced for large lattices)
    r_arr = np.arange(1, r_max + 1)
    corr[1:] *= np.exp(-GAMMA_DEC * r_arr / 4)
    
    mask = counts[1:] > 0
    r_filtered = r_arr[mask]
    corr_filtered = np.abs(corr[1:][mask])
    
    return r_filtered, corr_filtered


# ========== SINGLE LATTICE WORKER (FOR PARALLEL) ==========
def process_single_lattice(L_cur, seed_offset=0):
    """
    Worker function for parallel processing
    Runs complete FSS analysis for one lattice size
    
    Args:
        L_cur: Lattice size
        seed_offset: Random seed offset for reproducibility
    
    Returns:
        dict with eta_eff, cluster_stats, etc.
    """
    # Set unique seed per process
    np.random.seed(42 + seed_offset)
    
    # Setup process-specific logger
    logger = setup_logger(L_cur)
    
    print(f"[PID-{L_cur}] Starting analysis for L={L_cur}...")
    logger.info(f"=" * 50)
    logger.info(f"Starting lattice L = {L_cur}")
    logger.info(f"=" * 50)
    
    start_time = time.time()
    
    # Initialize
    spins = 2 * np.random.randint(0, 2, (L_cur, L_cur)) - 1
    kernel_phi = phi_kernel(L_cur)
    
    # Equilibration phase
    N_equil = max(2000, L_cur**2 // 2)
    cluster_sizes_equil = []
    
    print(f"[PID-{L_cur}] Equilibrating ({N_equil} steps)...")
    for step in range(N_equil):
        spins, _ = metropolis_step(spins, BETA_C, kernel_phi,
                                   G_YUK, THETA_TWIST)
        if step % 10 == 0:
            spins, cs = wolff_cluster(spins, BETA_C, logger=logger, 
                                     log_data=False)
            cluster_sizes_equil.append(cs)
    
    # Production phase
    N_steps_L = max(4000, L_cur**2 // 2)
    n_blocks = 16
    steps_per_block = N_steps_L // n_blocks
    
    blocks = []
    cluster_sizes_prod = []
    
    print(f"[PID-{L_cur}] Production run ({N_steps_L} steps, {n_blocks} blocks)...")
    
    for blk in range(n_blocks):
        # Block sampling
        for step in range(steps_per_block):
            spins, _ = metropolis_step(spins, BETA_C, kernel_phi,
                                      G_YUK, THETA_TWIST)
            
            # Wolff every 10 steps (with logging)
            if step % 10 == 0:
                spins, cs = wolff_cluster(spins, BETA_C, logger=logger,
                                         log_data=True)
                cluster_sizes_prod.append(cs)
        
        # Measure correlation
        r_max = L_cur // 4
        r_min, r_max_fit = max(8, L_cur // 32), L_cur // 4
        
        r, G = corr_2d(spins, r_max)
        mask_fit = (r >= r_min) & (r <= r_max_fit)
        
        if np.sum(mask_fit) >= 5:
            try:
                popt, _ = curve_fit(power_law, r[mask_fit], G[mask_fit],
                                   p0=[1.0, 0.8], maxfev=5000,
                                   bounds=([0.01, 0.1], [10.0, 3.0]))
                eta_blk = popt[1]
                
                if 0.3 < eta_blk < 2.0:
                    blocks.append(eta_blk)
                else:
                    blocks.append(np.nan)
            except:
                blocks.append(np.nan)
        else:
            blocks.append(np.nan)
        
        # Progress indicator
        if (blk + 1) % 4 == 0:
            print(f"[PID-{L_cur}] Progress: {(blk+1)/n_blocks*100:.0f}%")
    
    # Statistics
    blocks_clean = [b for b in blocks if not np.isnan(b)]
    if len(blocks_clean) > 0:
        eta_mean = np.mean(blocks_clean)
        eta_std = np.std(blocks_clean) / np.sqrt(len(blocks_clean))
    else:
        eta_mean, eta_std = np.nan, np.nan
    
    # Cluster statistics
    all_clusters = cluster_sizes_equil + cluster_sizes_prod
    cs_mean = np.mean(all_clusters)
    cs_std = np.std(all_clusters)
    cs_max = np.max(all_clusters)
    cs_frac_mean = cs_mean / (L_cur * L_cur)
    
    elapsed = time.time() - start_time
    
    print(f"[PID-{L_cur}] COMPLETE in {elapsed:.1f}s")
    print(f"[PID-{L_cur}]   η_eff = {eta_mean:.4f} ± {eta_std:.4f}")
    print(f"[PID-{L_cur}]   Cluster: mean={cs_mean:.1f}, max={cs_max}, "
          f"frac={cs_frac_mean:.4f}")
    
    logger.info(f"Lattice L={L_cur} summary:")
    logger.info(f"  η_eff = {eta_mean:.4f} ± {eta_std:.4f}")
    logger.info(f"  Clusters: mean={cs_mean:.1f}, max={cs_max}, "
               f"frac={cs_frac_mean:.4f}")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    
    return {
        'L': L_cur,
        'eta_mean': eta_mean,
        'eta_std': eta_std,
        'cluster_sizes': all_clusters,
        'runtime': elapsed
    }
    """
    Execute FSS with Wolff cluster tracking
    Focus: Large lattices for critical behavior
    """
    eta_effs = []
    eta_stds = []
    cluster_stats = {}  # {L: [sizes]}
    
    print("=" * 70)
    print("2D φ-Fixed CQFT: Wolff Cluster Analysis")
    print("Large Lattice Regime: L ∈ [128, 256, 512, 1024, 1536]")
    print("=" * 70)
    
    for L_idx, L_cur in enumerate(LS):
        print(f"\n[L = {L_cur}] Starting Wolff analysis...")
        logger.info(f"=" * 50)
        logger.info(f"Starting lattice L = {L_cur}")
        logger.info(f"=" * 50)
        
        # Initialize
        spins = 2 * np.random.randint(0, 2, (L_cur, L_cur)) - 1
        kernel_phi = phi_kernel(L_cur)
        
        # Equilibration phase (longer for large lattices)
        N_equil = max(2000, L_cur**2 // 2)
        cluster_sizes_equil = []
        
        print(f"  Equilibrating ({N_equil} steps)...")
        for step in range(N_equil):
            spins, _ = metropolis_step(spins, BETA_C, kernel_phi,
                                       G_YUK, THETA_TWIST)
            if step % 10 == 0:
                spins, cs = wolff_cluster(spins, BETA_C, log_data=False)
                cluster_sizes_equil.append(cs)
        
        # Production phase (with logging)
        N_steps_L = max(4000, L_cur**2 // 2)
        n_blocks = 16
        steps_per_block = N_steps_L // n_blocks
        
        blocks = []
        cluster_sizes_prod = []
        
        print(f"  Production run ({N_steps_L} steps, {n_blocks} blocks)...")
        
        for blk in range(n_blocks):
            # Block sampling
            for step in range(steps_per_block):
                spins, _ = metropolis_step(spins, BETA_C, kernel_phi,
                                          G_YUK, THETA_TWIST)
                
                # Wolff every 10 steps (with logging)
                if step % 10 == 0:
                    spins, cs = wolff_cluster(spins, BETA_C, log_data=True)
                    cluster_sizes_prod.append(cs)
            
            # Measure correlation
            r_max = L_cur // 4
            r_min, r_max_fit = max(8, L_cur // 32), L_cur // 4
            
            r, G = corr_2d(spins, r_max)
            mask_fit = (r >= r_min) & (r <= r_max_fit)
            
            if np.sum(mask_fit) >= 5:
                try:
                    popt, _ = curve_fit(power_law, r[mask_fit], G[mask_fit],
                                       p0=[1.0, 0.8], maxfev=5000,
                                       bounds=([0.01, 0.1], [10.0, 3.0]))
                    eta_blk = popt[1]
                    
                    if 0.3 < eta_blk < 2.0:
                        blocks.append(eta_blk)
                    else:
                        blocks.append(np.nan)
                except:
                    blocks.append(np.nan)
            else:
                blocks.append(np.nan)
        
        # Statistics
        blocks_clean = [b for b in blocks if not np.isnan(b)]
        if len(blocks_clean) > 0:
            eta_mean = np.mean(blocks_clean)
            eta_std = np.std(blocks_clean) / np.sqrt(len(blocks_clean))
        else:
            eta_mean, eta_std = np.nan, np.nan
        
        eta_effs.append(eta_mean)
        eta_stds.append(eta_std)
        
        # Store cluster statistics
        all_clusters = cluster_sizes_equil + cluster_sizes_prod
        cluster_stats[L_cur] = all_clusters
        
        # Cluster statistics
        cs_mean = np.mean(all_clusters)
        cs_std = np.std(all_clusters)
        cs_max = np.max(all_clusters)
        cs_frac_mean = cs_mean / (L_cur * L_cur)
        
        print(f"  η_eff = {eta_mean:.4f} ± {eta_std:.4f}")
        print(f"  Cluster stats:")
        print(f"    Mean size: {cs_mean:.1f} ± {cs_std:.1f}")
        print(f"    Max size: {cs_max}")
        print(f"    Mean fraction: {cs_frac_mean:.4f}")
        
        logger.info(f"Lattice L={L_cur} summary:")
        logger.info(f"  η_eff = {eta_mean:.4f} ± {eta_std:.4f}")
        logger.info(f"  Clusters: mean={cs_mean:.1f}, max={cs_max}, "
                   f"frac={cs_frac_mean:.4f}")
    
    return {
        'eta_effs': np.array(eta_effs),
        'eta_stds': np.array(eta_stds),
        'cluster_stats': cluster_stats,
        'LS': LS
    }


# ========== WOLFF VISUALIZATION ==========
def plot_wolff_analysis(results):
    """Comprehensive Wolff cluster visualization"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    cluster_stats = results['cluster_stats']
    LS_data = results['LS']
    
    # [0,0]: Cluster size distributions (all L)
    ax00 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(LS_data)))
    
    for idx, L in enumerate(LS_data):
        sizes = cluster_stats[L]
        bins = np.logspace(0, np.log10(max(sizes)), 50)
        hist, edges = np.histogram(sizes, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax00.loglog(centers, hist, 'o-', alpha=0.7, 
                   label=f'L={L}', color=colors[idx])
    
    ax00.set_xlabel('Cluster Size', fontsize=11)
    ax00.set_ylabel('P(size)', fontsize=11)
    ax00.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax00.legend(fontsize=9)
    ax00.grid(True, alpha=0.3)
    
    # [0,1]: Mean cluster size vs L (FSS scaling)
    ax01 = fig.add_subplot(gs[0, 1])
    mean_sizes = [np.mean(cluster_stats[L]) for L in LS_data]
    std_sizes = [np.std(cluster_stats[L]) for L in LS_data]
    
    ax01.errorbar(LS_data, mean_sizes, yerr=std_sizes, 
                 fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax01.set_xlabel('L', fontsize=11)
    ax01.set_ylabel('⟨Cluster Size⟩', fontsize=11)
    ax01.set_title('Mean Cluster Size Scaling', fontsize=12, fontweight='bold')
    ax01.set_xscale('log')
    ax01.set_yscale('log')
    ax01.grid(True, alpha=0.3)
    
    # Add power-law fit
    if len(LS_data) >= 3:
        log_L = np.log(LS_data)
        log_mean = np.log(mean_sizes)
        coeffs = np.polyfit(log_L, log_mean, 1)
        exponent = coeffs[0]
        L_fit = np.array(LS_data)
        fit_curve = np.exp(coeffs[1]) * L_fit**exponent
        ax01.plot(L_fit, fit_curve, '--', linewidth=2, 
                 label=f'~L^{exponent:.3f}', color='red')
        ax01.legend(fontsize=10)
    
    # [0,2]: Cluster fraction vs L
    ax02 = fig.add_subplot(gs[0, 2])
    fractions = [np.mean(cluster_stats[L]) / (L*L) for L in LS_data]
    ax02.plot(LS_data, fractions, 'o-', linewidth=2, markersize=8)
    ax02.set_xlabel('L', fontsize=11)
    ax02.set_ylabel('⟨Size⟩/L²', fontsize=11)
    ax02.set_title('Cluster Fraction', fontsize=12, fontweight='bold')
    ax02.set_xscale('log')
    ax02.grid(True, alpha=0.3)
    
    # [1,0-1]: Individual cluster histograms (linear scale)
    for idx, L in enumerate(LS_data[:2]):
        ax = fig.add_subplot(gs[1, idx])
        sizes = cluster_stats[L]
        ax.hist(sizes, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cluster Size', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'L={L} Distribution', fontsize=11, fontweight='bold')
        ax.axvline(np.mean(sizes), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean={np.mean(sizes):.1f}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    # [1,2]: Max cluster size vs L
    ax12 = fig.add_subplot(gs[1, 2])
    max_sizes = [np.max(cluster_stats[L]) for L in LS_data]
    ax12.plot(LS_data, max_sizes, 'o-', linewidth=2, markersize=8, color='purple')
    ax12.set_xlabel('L', fontsize=11)
    ax12.set_ylabel('Max Cluster Size', fontsize=11)
    ax12.set_title('Maximum Cluster Size', fontsize=12, fontweight='bold')
    ax12.set_xscale('log')
    ax12.set_yscale('log')
    ax12.grid(True, alpha=0.3)
    
    # [2,0]: η_eff vs L (FSS)
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.errorbar(LS_data, results['eta_effs'], yerr=results['eta_stds'],
                 fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax20.axhline(ETA_TARGET, ls='--', color='red', linewidth=2,
                label=f'Target η={ETA_TARGET}')
    ax20.set_xlabel('L', fontsize=11)
    ax20.set_ylabel('η_eff', fontsize=11)
    ax20.set_title('Anomalous Dimension FSS', fontsize=12, fontweight='bold')
    ax20.legend(fontsize=10)
    ax20.grid(True, alpha=0.3)
    ax20.set_ylim(0.5, 1.0)
    
    # [2,1]: Cluster size cumulative distribution
    ax21 = fig.add_subplot(gs[2, 1])
    for idx, L in enumerate(LS_data):
        sizes = np.sort(cluster_stats[L])
        cdf = np.arange(1, len(sizes) + 1) / len(sizes)
        ax21.semilogx(sizes, cdf, '-', alpha=0.7, 
                     label=f'L={L}', color=colors[idx], linewidth=2)
    ax21.set_xlabel('Cluster Size', fontsize=11)
    ax21.set_ylabel('CDF', fontsize=11)
    ax21.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax21.legend(fontsize=9)
    ax21.grid(True, alpha=0.3)
    
    # [2,2]: Summary statistics table
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis('off')
    
    summary_text = "WOLFF CLUSTER SUMMARY\n" + "="*35 + "\n\n"
    for L in LS_data:
        sizes = cluster_stats[L]
        summary_text += f"L = {L:4d}\n"
        summary_text += f"  Mean: {np.mean(sizes):8.1f}\n"
        summary_text += f"  Std:  {np.std(sizes):8.1f}\n"
        summary_text += f"  Max:  {np.max(sizes):8d}\n"
        summary_text += f"  Frac: {np.mean(sizes)/(L*L):8.4f}\n\n"
    
    summary_text += f"\nφ = {PHI:.6f}\n"
    summary_text += f"β_c = {BETA_C:.6f}\n"
    summary_text += f"Target η = {ETA_TARGET:.4f}\n"
    
    ax22.text(0.05, 0.95, summary_text, transform=ax22.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('2D φ-Fixed CQFT: Wolff Cluster Analysis', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('wolff_cluster_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: wolff_cluster_analysis.png")
    plt.show()


# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Wolff Cluster Analysis: Large Lattice Regime")
    print("This will take ~10-30 minutes depending on hardware")
    print("="*70 + "\n")
    
    results = run_wolff_fss_analysis()
    plot_wolff_analysis(results)
    
    print("\n✓ Analysis complete!")
    print("✓ Log file: wolff_cluster.log")
    print("✓ Plot: wolff_cluster_analysis.png")
    print("\n*** Wolff Cluster FSS Complete ***")
