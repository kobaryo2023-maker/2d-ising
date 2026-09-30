import numpy as np
import matplotlib.pyplot as plt

def run_2d_ising():
    L = 24  # 24 × 24 格子（全576スピン）
    temperatures = np.linspace(1.2, 3.5, 12)
    rng = np.random.default_rng(seed=42)
    
    # 記録用データ構造
    T_snapshots = [1.5, 2.27, 3.5]
    snapshots = {}
    mean_magnetizations = []
    
    print("シミュレーションを実行中...")
    for T in temperatures:
        spins = rng.choice([-1, 1], size=(L, L))
        m_list = []
        
        warmup = 500
        steps = 1500
        
        for sweep in range(warmup + steps):
            for _ in range(L * L):
                i = rng.integers(L)
                j = rng.integers(L)
                # 2次元（上下左右4つの近傍スピン）の和
                neighbors = (spins[(i-1)%L, j] + spins[(i+1)%L, j] +
                             spins[i, (j-1)%L] + spins[i, (j+1)%L])
                delta_E = 2.0 * spins[i, j] * neighbors
                
                if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                    spins[i, j] *= -1
            
            if sweep >= warmup:
                m_list.append(np.abs(np.mean(spins)))
        
        mean_magnetizations.append(np.mean(m_list))
        
        # 観察したい温度に近い状態を保持
        for T_target in T_snapshots:
            if abs(T - T_target) < 0.1 and T_target not in snapshots:
                snapshots[T_target] = spins.copy()

    # 描画処理
    fig = plt.figure(figsize=(12, 7))
    
    # 1. 磁化曲線のプロット（※ここで r を付与して修正）
    ax1 = fig.add_subplot(2, 3, (1, 3))
    ax1.plot(temperatures, mean_magnetizations, 'o-', color='crimson', label="Simulation")
    ax1.axvline(x=2.269, color='black', linestyle='--', label=r"Onsager $T_c \approx 2.269$")
    ax1.set_xlabel("Temperature T")
    ax1.set_ylabel("Absolute Magnetization per spin |M|")
    ax1.set_title("2D Ising Model Phase Transition")
    ax1.legend()
    ax1.grid(True)

    # 2. スピン状態の可視化（格子模様）
    for idx, T_target in enumerate(T_snapshots, start=4):
        ax = fig.add_subplot(2, 3, idx)
        ax.imshow(snapshots[T_target], cmap='binary')
        ax.set_title(f"T = {T_target:.2f}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

run_2d_ising()